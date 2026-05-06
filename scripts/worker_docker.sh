#!/usr/bin/env bash
# =============================================================================
# scripts/worker_docker.sh
#
# Constrói a imagem Docker do worker AsaMiia (CPU ou GPU).
#
# Uso:
#   ./scripts/worker_docker.sh --cpu --tag latest
#   ./scripts/worker_docker.sh --gpu --tag latest
#
# Opções:
#   --cpu           Constrói a imagem CPU  (target: runtime-cpu)
#   --gpu           Constrói a imagem GPU  (target: runtime-gpu)
#   --tag <tag>     Tag da imagem (padrão: latest)
#   --no-cache      Passa --no-cache ao docker build
#   --help          Exibe esta ajuda
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults — espelham as variáveis do Makefile
# ---------------------------------------------------------------------------
IMAGE_NAME="ml-inference-server"
TAG="latest"
MODE=""
NO_CACHE=""
DOCKERFILE="docker/Dockerfile.server"

# ---------------------------------------------------------------------------
# Cores (mesmas do Makefile)
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
info()    { echo -e "${BLUE}$*${NC}"; }
success() { echo -e "${GREEN}✓ $*${NC}"; }
warn()    { echo -e "${YELLOW}⚠ $*${NC}"; }
error()   { echo -e "${RED}✗ $*${NC}" >&2; }

usage() {
    echo "Uso: $0 [--cpu|--gpu] [--tag TAG] [--no-cache]"
    echo ""
    echo "Opções:"
    echo "  --cpu           Constrói imagem CPU  (target: runtime-cpu)"
    echo "  --gpu           Constrói imagem GPU  (target: runtime-gpu)"
    echo "  --tag TAG       Tag da imagem (padrão: latest)"
    echo "  --no-cache      Passa --no-cache ao docker build"
    echo "  --help          Exibe esta ajuda"
}

# ---------------------------------------------------------------------------
# Parse de argumentos
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)
            MODE="cpu"
            shift
            ;;
        --gpu)
            MODE="gpu"
            shift
            ;;
        --tag)
            TAG="${2:?'--tag requer um valor'}"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            error "Argumento desconhecido: $1"
            usage
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validação
# ---------------------------------------------------------------------------
if [[ -z "$MODE" ]]; then
    error "Especifique --cpu ou --gpu."
    usage
    exit 1
fi

if [[ ! -f "$DOCKERFILE" ]]; then
    error "Dockerfile não encontrado: $DOCKERFILE"
    error "Execute este script a partir da raiz do projeto."
    exit 1
fi

# ---------------------------------------------------------------------------
# Configuração por modo
# ---------------------------------------------------------------------------
if [[ "$MODE" == "cpu" ]]; then
    TARGET="runtime-cpu"
    ENABLE_GPU="False"
    FULL_TAG="${IMAGE_NAME}:${TAG}-cpu"
else
    TARGET="runtime-gpu"
    ENABLE_GPU="True"
    FULL_TAG="${IMAGE_NAME}:${TAG}-gpu"

    # Verifica suporte a GPU no host
    if ! docker info 2>/dev/null | grep -q "Runtimes.*nvidia\|nvidia.*Runtimes"; then
        warn "nvidia runtime não detectado — a imagem será construída, mas pode não rodar com GPU."
        warn "Instale: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    fi
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
info "========================================"
info "  AsaMiia Docker Build"
info "  Modo:       ${MODE^^}"
info "  Target:     $TARGET"
info "  Imagem:     $FULL_TAG"
info "  Dockerfile: $DOCKERFILE"
[[ -n "$NO_CACHE" ]] && info "  Cache:      DESABILITADO"
info "========================================"
echo ""

docker build \
    --file "$DOCKERFILE" \
    --target "$TARGET" \
    --tag "$FULL_TAG" \
    --build-arg ENABLE_GPU="$ENABLE_GPU" \
    --build-arg CONAN_BUILD_TYPE="Release" \
    --build-arg MESON_BUILDTYPE="release" \
    $NO_CACHE \
    .

echo ""
success "Imagem construída com sucesso: $FULL_TAG"
echo ""
info "Para executar:"
if [[ "$MODE" == "cpu" ]]; then
    echo "  docker run -d --rm -p 50052:50052 \\"
    echo "    -v \$(pwd)/models:/app/models:ro \\"
    echo "    $FULL_TAG"
else
    echo "  docker run -d --rm -p 50052:50052 --gpus all \\"
    echo "    -v \$(pwd)/models:/app/models:ro \\"
    echo "    $FULL_TAG"
fi
echo ""