#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char *argv[]) {
    char launcher_path[PATH_MAX];
    char runtime_path[PATH_MAX];

    if (realpath(argv[0], launcher_path) == NULL) {
        perror("MyGo could not resolve its launcher path");
        return 126;
    }

    char *last_slash = strrchr(launcher_path, '/');
    if (last_slash == NULL) {
        fputs("MyGo launcher is not inside an app bundle.\n", stderr);
        return 126;
    }
    *last_slash = '\0'; /* .../Contents/MacOS */

    last_slash = strrchr(launcher_path, '/');
    if (last_slash == NULL) {
        fputs("MyGo could not locate its Contents directory.\n", stderr);
        return 126;
    }
    *last_slash = '\0'; /* .../Contents */

    int runtime_length = snprintf(
        runtime_path,
        sizeof(runtime_path),
        "%s/Resources/MyGoRuntime/MyGo",
        launcher_path
    );
    if (runtime_length < 0 || (size_t)runtime_length >= sizeof(runtime_path)) {
        fputs("MyGo runtime path is too long.\n", stderr);
        return 126;
    }

    if (access(runtime_path, X_OK) != 0) {
        fprintf(stderr, "MyGo runtime is missing or not executable: %s\n", runtime_path);
        return 126;
    }

    char **runtime_argv = calloc((size_t)argc + 1, sizeof(char *));
    if (runtime_argv == NULL) {
        perror("MyGo could not allocate launcher arguments");
        return 126;
    }
    runtime_argv[0] = runtime_path;
    for (int index = 1; index < argc; index++) {
        runtime_argv[index] = argv[index];
    }

    execv(runtime_path, runtime_argv);
    fprintf(stderr, "MyGo could not start its runtime: %s\n", strerror(errno));
    free(runtime_argv);
    return 127;
}
