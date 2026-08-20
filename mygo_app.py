#!/usr/bin/env python3
"""Native macOS menu bar app for the MyGo search engine."""

from __future__ import annotations

import fcntl
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import objc
from AppKit import (
    NSApp,
    NSApplication,
    NSApplicationActivationPolicyRegular,
    NSBackingStoreBuffered,
    NSBezelStyleRounded,
    NSBox,
    NSButton,
    NSFont,
    NSImage,
    NSImageFramePhoto,
    NSImageScaleProportionallyUpOrDown,
    NSImageView,
    NSLineBreakByTruncatingTail,
    NSMenu,
    NSMenuItem,
    NSNoTitle,
    NSScrollView,
    NSSearchField,
    NSStatusBar,
    NSTextField,
    NSVariableStatusItemLength,
    NSView,
    NSViewHeightSizable,
    NSViewMinXMargin,
    NSViewMinYMargin,
    NSViewWidthSizable,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskMiniaturizable,
    NSWindowStyleMaskResizable,
    NSWindowStyleMaskTitled,
    NSWorkspace,
)
from Foundation import NSData, NSDistributedNotificationCenter, NSMakeRect, NSMakeSize, NSObject, NSURL
from PyObjCTools import AppHelper

from mygo_search import DATA_DIR, MyGoEngine, copy_image_to_clipboard, download_image


WINDOW_WIDTH = 780
WINDOW_HEIGHT = 650
RESULT_LIMIT = 12
GRID_GAP = 12
MIN_CARD_WIDTH = 320
MAX_GRID_COLUMNS = 3
CARD_SIDE_INSET = 10
CARD_TOP_INSET = 10
CARD_FOOTER_HEIGHT = 84
ACTIVATE_NOTIFICATION = "com.shilohlee.mygo.activate"


def acquire_instance_lock(data_dir=DATA_DIR):
    """Return a held file lock, or None when another MyGo process owns it."""
    data_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (data_dir / "MyGo.lock").open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return None
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"{os.getpid()}\n")
    lock_file.flush()
    return lock_file


def notify_running_instance():
    """Ask the existing process to reveal its search window."""
    NSDistributedNotificationCenter.defaultCenter().postNotificationName_object_(
        ACTIVATE_NOTIFICATION,
        None,
    )


class FlippedView(NSView):
    """Make manually positioned result rows grow from the top down."""

    def isFlipped(self):
        return True


def label(text, frame, font_size=13, bold=False):
    field = NSTextField.labelWithString_(text)
    field.setFrame_(frame)
    field.setFont_(NSFont.boldSystemFontOfSize_(font_size) if bold else NSFont.systemFontOfSize_(font_size))
    field.setLineBreakMode_(NSLineBreakByTruncatingTail)
    return field


def result_grid_metrics(width, item_count):
    """Return responsive grid geometry for the available result width."""
    columns = max(
        1,
        min(
            MAX_GRID_COLUMNS,
            math.floor((width + GRID_GAP) / (MIN_CARD_WIDTH + GRID_GAP)),
        ),
    )
    card_width = (width - GRID_GAP * (columns - 1)) / columns
    image_width = card_width - CARD_SIDE_INSET * 2
    image_height = image_width * 9 / 16
    card_height = image_height + CARD_FOOTER_HEIGHT + CARD_TOP_INSET
    row_count = math.ceil(item_count / columns) if item_count else 0
    content_height = (
        row_count * card_height + max(row_count - 1, 0) * GRID_GAP
    )
    return columns, card_width, card_height, image_width, image_height, content_height


class MyGoAppDelegate(NSObject):
    def applicationDidFinishLaunching_(self, _notification):
        self.model_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mygo-model")
        self.io_executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="mygo-io")
        self.engine = MyGoEngine(status=self._engine_status)
        self.results = []
        self.result_cards = []
        self.image_views = []
        self.results_generation = 0
        self.notification_center = NSDistributedNotificationCenter.defaultCenter()
        self.notification_center.addObserver_selector_name_object_(
            self,
            "showWindowFromNotification:",
            ACTIVATE_NOTIFICATION,
            None,
        )
        self._build_main_menu()
        self._build_status_item()
        self._build_window()
        self.showWindow_(None)
        self.model_executor.submit(self._load_engine)

    def applicationShouldTerminateAfterLastWindowClosed_(self, _application):
        return False

    def applicationWillTerminate_(self, _notification):
        self.notification_center.removeObserver_(self)
        self.model_executor.shutdown(wait=False, cancel_futures=True)
        self.io_executor.shutdown(wait=False, cancel_futures=True)

    def applicationShouldHandleReopen_hasVisibleWindows_(self, _application, _has_visible_windows):
        self.showWindow_(None)
        return True

    @objc.python_method
    def _build_main_menu(self):
        main_menu = NSMenu.alloc().initWithTitle_("Main Menu")
        app_menu_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "MyGo",
            None,
            "",
        )
        main_menu.addItem_(app_menu_item)

        app_menu = NSMenu.alloc().initWithTitle_("MyGo")
        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Quit MyGo",
            "terminate:",
            "q",
        )
        app_menu.addItem_(quit_item)
        app_menu_item.setSubmenu_(app_menu)
        NSApp.setMainMenu_(main_menu)

    @objc.python_method
    def _build_status_item(self):
        self.status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(NSVariableStatusItemLength)
        self.status_item.button().setTitle_("MyGO")
        self.status_item.button().setToolTip_("Search MyGo images")

        menu = NSMenu.alloc().init()

        open_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Open MyGo Search", "showWindow:", ""
        )
        open_item.setTarget_(self)
        menu.addItem_(open_item)

        data_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Open Data Folder", "openDataFolder:", ""
        )
        data_item.setTarget_(self)
        menu.addItem_(data_item)

        login_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(
            "Open Login Items Settings…", "openLoginItems:", ""
        )
        login_item.setTarget_(self)
        menu.addItem_(login_item)

        menu.addItem_(NSMenuItem.separatorItem())
        quit_item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_("Quit MyGo", "terminate:", "q")
        menu.addItem_(quit_item)
        self.status_item.setMenu_(menu)

    @objc.python_method
    def _build_window(self):
        style = (
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable
            | NSWindowStyleMaskResizable
        )
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT),
            style,
            NSBackingStoreBuffered,
            False,
        )
        self.window.setTitle_("MyGo Search")
        self.window.setMinSize_(NSMakeSize(620, 480))
        self.window.setReleasedWhenClosed_(False)
        self.window.setDelegate_(self)
        self.window.center()

        content = self.window.contentView()

        self.search_field = NSSearchField.alloc().initWithFrame_(
            NSMakeRect(20, WINDOW_HEIGHT - 56, WINDOW_WIDTH - 135, 30)
        )
        self.search_field.setPlaceholderString_(
            "Hybrid search · /s semantic only · /f fuzzy only"
        )
        self.search_field.setTarget_(self)
        self.search_field.setAction_("performSearch:")
        self.search_field.setSendsWholeSearchString_(True)
        self.search_field.setSendsSearchStringImmediately_(False)
        self.search_field.setEnabled_(False)
        self.search_field.setAutoresizingMask_(NSViewWidthSizable | NSViewMinYMargin)
        content.addSubview_(self.search_field)

        self.search_button = NSButton.alloc().initWithFrame_(
            NSMakeRect(WINDOW_WIDTH - 105, WINDOW_HEIGHT - 56, 85, 30)
        )
        self.search_button.setTitle_("Search")
        self.search_button.setBezelStyle_(NSBezelStyleRounded)
        self.search_button.setTarget_(self)
        self.search_button.setAction_("performSearch:")
        self.search_button.setEnabled_(False)
        self.search_button.setAutoresizingMask_(NSViewMinXMargin | NSViewMinYMargin)
        content.addSubview_(self.search_button)

        self.status_label = label("Starting MyGo…", NSMakeRect(22, WINDOW_HEIGHT - 82, WINDOW_WIDTH - 44, 20), 12)
        self.status_label.setTextColor_(self.status_label.textColor().colorWithAlphaComponent_(0.72))
        self.status_label.setAutoresizingMask_(NSViewWidthSizable | NSViewMinYMargin)
        content.addSubview_(self.status_label)

        self.scroll_view = NSScrollView.alloc().initWithFrame_(
            NSMakeRect(20, 20, WINDOW_WIDTH - 40, WINDOW_HEIGHT - 112)
        )
        self.scroll_view.setHasVerticalScroller_(True)
        self.scroll_view.setAutohidesScrollers_(True)
        self.scroll_view.setAutoresizingMask_(NSViewWidthSizable | NSViewHeightSizable)

        self.results_view = FlippedView.alloc().initWithFrame_(
            NSMakeRect(0, 0, WINDOW_WIDTH - 42, WINDOW_HEIGHT - 114)
        )
        self.results_view.setAutoresizingMask_(NSViewWidthSizable)
        self.scroll_view.setDocumentView_(self.results_view)
        content.addSubview_(self.scroll_view)

    def windowDidResize_(self, _notification):
        self._layout_results()

    @objc.IBAction
    def showWindow_(self, _sender):
        self.window.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)
        if self.engine.is_ready:
            self.window.makeFirstResponder_(self.search_field)

    def showWindowFromNotification_(self, _notification):
        self.showWindow_(None)

    @objc.IBAction
    def openDataFolder_(self, _sender):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        NSWorkspace.sharedWorkspace().openURL_(NSURL.fileURLWithPath_(str(DATA_DIR)))

    @objc.IBAction
    def openLoginItems_(self, _sender):
        url = NSURL.URLWithString_("x-apple.systempreferences:com.apple.LoginItems-Settings.extension")
        NSWorkspace.sharedWorkspace().openURL_(url)

    @objc.python_method
    def _engine_status(self, message):
        AppHelper.callAfter(self._set_status, message)

    @objc.python_method
    def _set_status(self, message):
        self.status_label.setStringValue_(message)

    @objc.python_method
    def _load_engine(self):
        try:
            self.engine.load()
        except Exception as exc:
            AppHelper.callAfter(self._engine_failed, str(exc))
            return
        AppHelper.callAfter(self._engine_ready)

    @objc.python_method
    def _engine_ready(self):
        self.search_field.setEnabled_(True)
        self.search_button.setEnabled_(True)
        self._set_status("Ready · model stays loaded while MyGo is running")
        self.window.makeFirstResponder_(self.search_field)

    @objc.python_method
    def _engine_failed(self, message):
        self._set_status(f"Could not start: {message}")

    @objc.IBAction
    def performSearch_(self, _sender):
        query = self.search_field.stringValue().strip()
        if not query or not self.engine.is_ready:
            return
        self.search_field.setEnabled_(False)
        self.search_button.setEnabled_(False)
        self._set_status(f'Searching for “{query}”…')
        self.model_executor.submit(self._search, query)

    @objc.python_method
    def _search(self, query):
        try:
            results = self.engine.search(query, top_n=RESULT_LIMIT)
        except Exception as exc:
            AppHelper.callAfter(self._search_failed, str(exc))
            return
        AppHelper.callAfter(self._show_results, query, results)

    @objc.python_method
    def _search_failed(self, message):
        self.search_field.setEnabled_(True)
        self.search_button.setEnabled_(True)
        self._set_status(f"Search failed: {message}")

    @objc.python_method
    def _show_results(self, query, results):
        self.results = results
        self.results_generation += 1
        generation = self.results_generation
        self.result_cards = []
        self.image_views = []
        for child in list(self.results_view.subviews()):
            child.removeFromSuperview()

        for index, (score, image) in enumerate(results):
            card = self._make_result_card(index, score, image)
            self.result_cards.append(card)
            self.results_view.addSubview_(card["box"])
            self.io_executor.submit(self._load_thumbnail, generation, index, image["url"])

        self._layout_results()

        self.search_field.setEnabled_(True)
        self.search_button.setEnabled_(True)
        self._set_status(f'{len(results)} results for “{query}” · click Copy Image')
        self.window.makeFirstResponder_(self.search_field)

    @objc.python_method
    def _make_result_card(self, index, score, image):
        box = NSBox.alloc().initWithFrame_(NSMakeRect(0, 0, MIN_CARD_WIDTH, 280))
        box.setTitlePosition_(NSNoTitle)

        image_view = NSImageView.alloc().initWithFrame_(NSMakeRect(0, 0, 1, 1))
        image_view.setImageFrameStyle_(NSImageFramePhoto)
        image_view.setImageScaling_(NSImageScaleProportionallyUpOrDown)
        box.addSubview_(image_view)
        self.image_views.append(image_view)

        title = label(image["alt"], NSMakeRect(0, 0, 1, 1), 15, True)
        title.setToolTip_(image["alt"])
        box.addSubview_(title)

        details = f'Episode {image["episode"]}  ·  score {score:.3f}  ·  popularity {image["popularity"]}'
        details_label = label(details, NSMakeRect(0, 0, 1, 1), 11)
        box.addSubview_(details_label)

        button = NSButton.alloc().initWithFrame_(NSMakeRect(0, 0, 112, 32))
        button.setTitle_("Copy Image")
        button.setBezelStyle_(NSBezelStyleRounded)
        button.setTag_(index)
        button.setTarget_(self)
        button.setAction_("copyResult:")
        box.addSubview_(button)
        return {
            "box": box,
            "image": image_view,
            "title": title,
            "details": details_label,
            "button": button,
        }

    @objc.python_method
    def _layout_results(self):
        if not hasattr(self, "scroll_view") or not hasattr(self, "result_cards"):
            return
        width = max(float(self.scroll_view.contentSize().width), 480)
        (
            columns,
            card_width,
            card_height,
            image_width,
            image_height,
            content_height,
        ) = result_grid_metrics(width, len(self.result_cards))
        viewport_height = float(self.scroll_view.contentSize().height)
        self.results_view.setFrameSize_(
            NSMakeSize(width, max(content_height, viewport_height))
        )

        for index, card in enumerate(self.result_cards):
            column = index % columns
            row = index // columns
            x = column * (card_width + GRID_GAP)
            y = row * (card_height + GRID_GAP)
            card["box"].setFrame_(NSMakeRect(x, y, card_width, card_height))
            card["image"].setFrame_(
                NSMakeRect(
                    CARD_SIDE_INSET,
                    CARD_FOOTER_HEIGHT,
                    image_width,
                    image_height,
                )
            )
            card["title"].setFrame_(
                NSMakeRect(CARD_SIDE_INSET, 52, image_width, 24)
            )
            card["details"].setFrame_(
                NSMakeRect(CARD_SIDE_INSET, 25, max(card_width - 146, 100), 20)
            )
            card["button"].setFrame_(
                NSMakeRect(card_width - 122, 14, 112, 32)
            )

    @objc.python_method
    def _load_thumbnail(self, generation, index, url):
        try:
            data = download_image(url)
        except Exception:
            return
        AppHelper.callAfter(self._set_thumbnail, generation, index, data)

    @objc.python_method
    def _set_thumbnail(self, generation, index, raw_data):
        if generation != self.results_generation or index >= len(self.image_views):
            return
        data = NSData.dataWithBytes_length_(raw_data, len(raw_data))
        image = NSImage.alloc().initWithData_(data)
        if image is not None:
            self.image_views[index].setImage_(image)

    @objc.IBAction
    def copyResult_(self, sender):
        index = sender.tag()
        if index < 0 or index >= len(self.results):
            return
        image = self.results[index][1]
        sender.setEnabled_(False)
        self._set_status(f'Copying “{image["alt"]}”…')
        self.io_executor.submit(self._copy_result, image, sender)

    @objc.python_method
    def _copy_result(self, image, button):
        try:
            copy_image_to_clipboard(image["url"])
        except Exception as exc:
            AppHelper.callAfter(self._copy_finished, button, f"Copy failed: {exc}")
            return
        AppHelper.callAfter(self._copy_finished, button, f'Copied “{image["alt"]}” to clipboard')

    @objc.python_method
    def _copy_finished(self, button, message):
        button.setEnabled_(True)
        self._set_status(message)


def main():
    if sys.platform != "darwin":
        raise SystemExit("MyGo.app is only supported on macOS")
    if "--self-test" in sys.argv:
        engine = MyGoEngine(status=None).load()
        if not engine.search("大家冷靜", top_n=1):
            raise SystemExit("MyGo self-test returned no search results")
        return
    instance_lock = acquire_instance_lock()
    if instance_lock is None:
        notify_running_instance()
        return
    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(NSApplicationActivationPolicyRegular)
    delegate = MyGoAppDelegate.alloc().init()
    delegate.instance_lock = instance_lock
    app.setDelegate_(delegate)
    app.run()


if __name__ == "__main__":
    main()
