#!/usr/bin/env bash
# ============================================
# Build Worker Docker Image (Meson + Conan)
# ============================================
set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
IMAGE_NAME="ml-inference-worker"
IMAGE_TAG="latest"
ENABLE_GPU="False"
BUILD_TYPE="release"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)        ENABLE_GPU="True";  shift ;;
        --cpu)        ENABLE_GPU="False"; shift ;;
        --debug)      BUILD_TYPE="debug"; shift ;;
        --release)    BUILD_TYPE="release"; shift ;;
        --tag)        IMAGE_TAG="$2";     shift 2 ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cpu                CPU build (default)"
            echo "  --gpu                GPU build (CUDA)"
            echo "  --debug              Debug build type"
            echo "  --release            Release build type (default)"
            echo "  --tag TAG            Docker image tag (default: latest)"
            echo "  --help, -h           Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 --cpu"
            echo "  $0 --gpu --tag v1.0.0"
            echo "  $0 --debug"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Update tag with variant
if [ "$ENABLE_GPU" = "True" ]; then
    IMAGE_TAG="${IMAGE_TAG}-gpu"
    RUNTIME_STAGE="runtime-gpu"
else
    IMAGE_TAG="${IMAGE_TAG}-cpu"
    RUNTIME_STAGE="runtime-cpu"
fi

FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Building Worker Docker Image (Meson)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Image:       ${FULL_IMAGE_NAME}"
echo "  Runtime:     ${RUNTIME_STAGE}"
echo "  GPU:         ${ENABLE_GPU}"
echo "  Build Type:  ${BUILD_TYPE}"
echo ""

# Check Dockerfile
if [ ! -f "./docker/Dockerfile.worker" ]; then
    echo -e "${RED}Error: ./docker/Dockerfile.worker not found${NC}"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check required files
echo -e "${BLUE}Checking required files...${NC}"
required_files=(
    "conanfile.py"
    "meson.build"
    "meson_options.txt"
    "asa-poc-miia.pc.in"
    "proto/common.proto"
    "proto/worker.proto"
    "cpp/worker/src/main.cpp"
    "cpp/worker/src/worker_server.cpp"
    "cpp/worker/src/inference_engine.cpp"
    "cpp/worker/meson.build"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}✗ Missing: $file${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ Found: $file${NC}"
done
echo ""

# Build
echo -e "${BLUE}Building Docker image...${NC}"
echo -e "${YELLOW}This may take several minutes on first build...${NC}"
echo ""

DOCKER_BUILDKIT=1 docker build \
    --file ./docker/Dockerfile.worker \
    --target ${RUNTIME_STAGE} \
    --build-arg ENABLE_GPU=${ENABLE_GPU} \
    --build-arg BUILD_TYPE=${BUILD_TYPE} \
    --tag ${FULL_IMAGE_NAME} \
    --progress=plain \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Build Successful!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Image: ${FULL_IMAGE_NAME}"
    echo ""
    docker images ${IMAGE_NAME} --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo ""
    echo "1. Test:"
    echo "   docker run --rm ${FULL_IMAGE_NAME} --help"
    echo ""
    echo "2. Run:"
    if [ "$ENABLE_GPU" = "True" ]; then
        echo "   docker run --rm --gpus all -p 50052:50052 -v \$(pwd)/models:/app/models:ro ${FULL_IMAGE_NAME}"
    else
        echo "   docker run --rm -p 50052:50052 -v \$(pwd)/models:/app/models:ro ${FULL_IMAGE_NAME}"
    fi
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}✗ Build Failed${NC}"
    echo -e "${RED}========================================${NC}"
    exit 1
fi