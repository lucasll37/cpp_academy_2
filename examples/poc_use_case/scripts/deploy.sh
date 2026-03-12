set -x
set -e

#
# sanity check: number of arguments
#

if [[ $# -ne 0 ]]; then
    echo "Expected exactly 0 parameters." >&2
    exit 1
fi

echo "Building the current package in debug and release modes"

conan create ./ --build=missing --profile=asa-debug
conan create ./ --build=missing --profile=asa-release

echo "Uploading local packages to registry"

conan upload "*" --remote=asa-libs --confirm
