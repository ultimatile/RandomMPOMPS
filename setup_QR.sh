#!/bin/bash
cd "$(dirname "$0")"
BUILD_DIR="code/tensornetwork/build"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake ..
make