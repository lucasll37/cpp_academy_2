"""
Script to create ONNX models for testing.
Downloads PyTorch pretrained models and converts them to ONNX format.
"""

from __future__ import annotations

import os
import sys
import numpy as np


# =============================================================================
# Terminal output
# =============================================================================

class Colors:
    BLUE   = '\033[0;34m'
    GREEN  = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED    = '\033[0;31m'
    NC     = '\033[0m'


def print_section(text: str) -> None:
    print(f"\n{Colors.BLUE}{'='*60}{Colors.NC}")
    print(f"{Colors.BLUE}{text}{Colors.NC}")
    print(f"{Colors.BLUE}{'='*60}{Colors.NC}\n")

def print_success(text: str) -> None:
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")

def print_error(text: str) -> None:
    print(f"{Colors.RED}✗ {text}{Colors.NC}")

def print_info(text: str) -> None:
    print(f"  {text}")

def print_warning(text: str) -> None:
    print(f"{Colors.YELLOW}⚠ {text}{Colors.NC}")


# =============================================================================
# Dependency checks
# =============================================================================

def check_dependencies() -> None:
    """Abort if required packages are missing."""
    missing = []
    for pkg in ('onnx', 'numpy', 'onnxscript'):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print_section("ERROR: Missing Dependencies")
        print_error(f"Required packages not found: {', '.join(missing)}")
        print()
        print(f"  pip install {' '.join(missing)}")
        print()
        print("Or reinstall all requirements:")
        print("  pip install -r python/requirements.txt")
        sys.exit(1)


def check_pytorch() -> bool:
    """Return True if PyTorch + torchvision are available."""
    try:
        import torch
        import torchvision
        print_success(f"PyTorch {torch.__version__} detected")
        print_info(f"Torchvision {torchvision.__version__}")
        return True
    except ImportError:
        print_error("PyTorch not installed")
        print()
        print("PyTorch is REQUIRED to download and convert pretrained models.")
        print()
        print("Install options:")
        print("  CPU only (~800 MB):")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print()
        print("  GPU CUDA 11.8 (~2 GB):")
        print("    pip install torch torchvision")
        print()
        print("  GPU CUDA 12.1+ (~2 GB):")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        return False


# =============================================================================
# Simple test model builders
# =============================================================================

def create_simple_linear_model():
    """Build a minimal ONNX model: output = input * 2 + 1."""
    import onnx
    from onnx import helper, TensorProto

    scale_tensor = helper.make_tensor('scale', TensorProto.FLOAT, [1], [2.0])
    bias_tensor  = helper.make_tensor('bias',  TensorProto.FLOAT, [1], [1.0])

    graph = helper.make_graph(
        nodes=[
            helper.make_node('Mul', inputs=['input', 'scale'], outputs=['scaled']),
            helper.make_node('Add', inputs=['scaled', 'bias'],  outputs=['output']),
        ],
        name='simple_linear',
        inputs=[helper.make_tensor_value_info('input',  TensorProto.FLOAT, [1, 5])],
        outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 5])],
        initializer=[scale_tensor, bias_tensor],
    )

    model = helper.make_model(graph, producer_name='miia')
    model.opset_import[0].version = 13
    onnx.checker.check_model(model)
    return model


def create_classification_model():
    """Build a minimal ONNX classifier: Linear(4→3) + Softmax."""
    import onnx
    from onnx import helper, TensorProto

    weights = np.random.randn(4, 3).astype(np.float32) * 0.1
    bias    = np.zeros(3, dtype=np.float32)

    weights_tensor = helper.make_tensor('weights', TensorProto.FLOAT, [4, 3], weights.flatten().tolist())
    bias_tensor    = helper.make_tensor('bias',    TensorProto.FLOAT, [3],    bias.tolist())

    graph = helper.make_graph(
        nodes=[
            helper.make_node('MatMul',  inputs=['input', 'weights'], outputs=['logits_raw']),
            helper.make_node('Add',     inputs=['logits_raw', 'bias'], outputs=['logits']),
            helper.make_node('Softmax', inputs=['logits'],             outputs=['output'], axis=1),
        ],
        name='simple_classifier',
        inputs=[helper.make_tensor_value_info('input',  TensorProto.FLOAT, [1, 4])],
        outputs=[helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])],
        initializer=[weights_tensor, bias_tensor],
    )

    model = helper.make_model(graph, producer_name='miia')
    model.opset_import[0].version = 13
    onnx.checker.check_model(model)
    return model


# =============================================================================
# PyTorch → ONNX conversion helpers
# =============================================================================

# Registry: model_name → (loader_fn, input_shape)
_PYTORCH_MODELS: dict[str, dict] = {
    'resnet18': {
        'label':       '3. ResNet-18',
        'description': 'ImageNet classifier (~44 MB)',
        'filename':    'resnet18.onnx',
    },
    'mobilenet_v2': {
        'label':       '4. MobileNet V2',
        'description': 'Lightweight classifier (~14 MB)',
        'filename':    'mobilenet_v2.onnx',
    },
    'squeezenet1_0': {
        'label':       '5. SqueezeNet 1.0',
        'description': 'Very lightweight classifier (~5 MB)',
        'filename':    'squeezenet1_0.onnx',
    },
}

_INPUT_SHAPE = (1, 3, 224, 224)


def _load_torchvision_model(model_name: str):
    """Instantiate a pretrained torchvision model by name."""
    import torchvision.models as tv

    loaders = {
        'resnet18':      lambda: tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1),
        'resnet50':      lambda: tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V1),
        'mobilenet_v2':  lambda: tv.mobilenet_v2(weights=tv.MobileNet_V2_Weights.IMAGENET1K_V1),
        'squeezenet1_0': lambda: tv.squeezenet1_0(weights=tv.SqueezeNet1_0_Weights.IMAGENET1K_V1),
        'efficientnet_b0': lambda: tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1),
    }

    if model_name not in loaders:
        raise ValueError(f"Unknown model: '{model_name}'. Available: {list(loaders)}")

    return loaders[model_name]()


def _consolidate_external_data(model_path: str) -> None:
    """Merge a sidecar .data file back into the .onnx file if present.

    torch.onnx.export() may create <model>.onnx.data for large models.
    This function folds it back into a single self-contained file.
    """
    import onnx

    data_file = model_path + '.data'
    if not os.path.exists(data_file):
        return

    print_info(f"Found external data: {os.path.basename(data_file)}")
    print_info("Consolidating into single file...")

    try:
        model = onnx.load(model_path, load_external_data=True)
        onnx.save(model, model_path, save_as_external_data=False)
        os.remove(data_file)
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print_success(f"Consolidated ({size_mb:.1f} MB)")
    except Exception as exc:
        print_warning(f"Could not consolidate: {exc}")
        print_info("Both .onnx and .data files will be required at runtime.")


def convert_pytorch_to_onnx(model_name: str, output_path: str) -> bool:
    """Download a pretrained torchvision model and export it as ONNX."""
    import torch
    import onnx

    print_info("Downloading pretrained weights...")

    try:
        model = _load_torchvision_model(model_name)
        print_success("Pretrained weights downloaded")
        model.eval()

        print_info("Converting to ONNX...")
        dummy_input = torch.randn(*_INPUT_SHAPE)

        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
            opset_version=13,
            do_constant_folding=True,
            export_params=True,
        )
        print_success("Converted to ONNX")

        print_info("Validating...")
        onnx.checker.check_model(onnx.load(output_path))
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print_success(f"Validated ({size_mb:.1f} MB) → {output_path}")

        _consolidate_external_data(output_path)
        return True

    except Exception as exc:
        print_error(f"Failed: {exc}")
        print()
        print_info("Common causes:")
        print_info("  pip install onnxscript")
        print_info("  pip install --upgrade torch torchvision")
        print_info("  Try a smaller model if running out of memory")
        return False


# =============================================================================
# Model creation tasks
# =============================================================================

def _save_simple_model(output_dir: str, label: str, filename: str,
                       builder_fn, input_spec: str, output_spec: str,
                       import_onnx) -> bool:
    """Create, save, and report a simple ONNX model. Returns success flag."""
    print(f"{Colors.YELLOW}{label}{Colors.NC}")
    try:
        model = builder_fn()
        path  = os.path.join(output_dir, filename)
        import_onnx.save(model, path)
        size_kb = os.path.getsize(path) / 1024
        print_success(f"Created ({size_kb:.1f} KB)")
        print_info(f"Input:  {input_spec}")
        print_info(f"Output: {output_spec}")
        return True
    except Exception as exc:
        print_error(f"Failed: {exc}")
        return False


def _create_simple_models(output_dir: str, import_onnx) -> tuple[int, int]:
    """Build the two hand-crafted test models. Returns (successes, total)."""
    tasks = [
        {
            'label':       '1. Simple Linear Model (output = input × 2 + 1)',
            'filename':    'simple_linear.onnx',
            'builder':     create_simple_linear_model,
            'input_spec':  "'input'  [1, 5] float32",
            'output_spec': "'output' [1, 5] float32",
        },
        {
            'label':       '2. Simple Classification Model',
            'filename':    'simple_classifier.onnx',
            'builder':     create_classification_model,
            'input_spec':  "'input'  [1, 4] float32",
            'output_spec': "'output' [1, 3] float32",
        },
    ]

    successes = 0
    for t in tasks:
        print()
        ok = _save_simple_model(
            output_dir, t['label'], t['filename'], t['builder'],
            t['input_spec'], t['output_spec'], import_onnx,
        )
        if ok:
            successes += 1

    return successes, len(tasks)


def _create_pytorch_models(output_dir: str) -> tuple[int, int]:
    """Convert pretrained torchvision models to ONNX. Returns (successes, total)."""
    print_section("Converting Pretrained Models from PyTorch")

    successes = 0
    for name, info in _PYTORCH_MODELS.items():
        print(f"\n{Colors.YELLOW}{info['label']}{Colors.NC}")
        print_info(f"Description: {info['description']}")
        print_info(f"Input:  [1, 3, 224, 224] (RGB image)")
        print_info(f"Output: [1, 1000] (ImageNet classes)")

        output_path = os.path.join(output_dir, info['filename'])
        if convert_pytorch_to_onnx(name, output_path):
            successes += 1

    return successes, len(_PYTORCH_MODELS)


# =============================================================================
# Summary & usage
# =============================================================================

def _print_model_listing(output_dir: str) -> None:
    if not os.path.exists(output_dir):
        return

    files = sorted(f for f in os.listdir(output_dir) if f.endswith('.onnx'))
    if not files:
        print(f"  {Colors.YELLOW}No models found{Colors.NC}")
        return

    for filename in files:
        size_mb  = os.path.getsize(os.path.join(output_dir, filename)) / (1024 * 1024)
        size_str = f"{size_mb:.1f} MB" if size_mb >= 1 else f"{size_mb * 1024:.1f} KB"
        print(f"  {Colors.GREEN}✓{Colors.NC} {filename:<30} ({size_str})")


def _print_usage() -> None:
    print_section("Usage Instructions")
    print("Test the models:")
    print()
    print(f"  {Colors.BLUE}# Terminal 1 — start worker{Colors.NC}")
    print(f"  make run-worker")
    print()
    print(f"  {Colors.BLUE}# Terminal 2 — run client{Colors.NC}")
    print(f"  make run-client")


def _print_final_status(success: int, total: int) -> None:
    if success == total:
        print(f"{Colors.GREEN}{'='*60}{Colors.NC}")
        print(f"{Colors.GREEN}✓ All {total} models created successfully!{Colors.NC}")
        print(f"{Colors.GREEN}{'='*60}{Colors.NC}")
    else:
        failed = total - success
        print(f"{Colors.YELLOW}{'='*60}{Colors.NC}")
        print(f"{Colors.YELLOW}⚠ {failed}/{total} model(s) failed{Colors.NC}")
        print(f"{Colors.YELLOW}{'='*60}{Colors.NC}")


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    check_dependencies()

    import onnx  # safe after check_dependencies()

    print_section("ML Inference — ONNX Model Creation")

    # PyTorch availability
    print("Checking PyTorch installation...")
    pytorch_available = check_pytorch()
    print()

    if not pytorch_available:
        print_warning("PyTorch not found — only simple test models will be created.")
        print()
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("\nInstall PyTorch and run again:")
            print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
            sys.exit(1)

    output_dir = 'models'
    os.makedirs(output_dir, exist_ok=True)

    # ── Simple models ──────────────────────────────────────────────────────
    print_section("Creating ONNX Models")
    s1, t1 = _create_simple_models(output_dir, onnx)

    # ── PyTorch models ─────────────────────────────────────────────────────
    s2, t2 = (0, 0)
    if pytorch_available:
        s2, t2 = _create_pytorch_models(output_dir)

    # ── Summary ────────────────────────────────────────────────────────────
    total_success = s1 + s2
    total_count   = t1 + t2

    print_section("Summary")
    print(f"Models created: {Colors.GREEN}{total_success}/{total_count}{Colors.NC}\n")
    print("Available models:")
    _print_model_listing(output_dir)
    print()

    _print_usage()
    _print_final_status(total_success, total_count)


if __name__ == '__main__':
    main()

# """
# Script to create ONNX models for testing.
# Downloads PyTorch pretrained models and converts them to ONNX format.
# """

# import numpy as np
# import sys
# import os

# # Color codes for terminal output
# class Colors:
#     BLUE = '\033[0;34m'
#     GREEN = '\033[0;32m'
#     YELLOW = '\033[1;33m'
#     RED = '\033[0;31m'
#     NC = '\033[0m'

# def print_section(text):
#     print(f"\n{Colors.BLUE}{'='*60}{Colors.NC}")
#     print(f"{Colors.BLUE}{text}{Colors.NC}")
#     print(f"{Colors.BLUE}{'='*60}{Colors.NC}\n")

# def print_success(text):
#     print(f"{Colors.GREEN}✓ {text}{Colors.NC}")

# def print_error(text):
#     print(f"{Colors.RED}✗ {text}{Colors.NC}")

# def print_info(text):
#     print(f"  {text}")

# def print_warning(text):
#     print(f"{Colors.YELLOW}⚠ {text}{Colors.NC}")

# def check_dependencies():
#     """Check if required packages are installed."""
#     missing = []
    
#     try:
#         import onnx
#     except ImportError:
#         missing.append('onnx')
    
#     try:
#         import numpy
#     except ImportError:
#         missing.append('numpy')
    
#     try:
#         import onnxscript
#     except ImportError:
#         missing.append('onnxscript')
    
#     if missing:
#         print_section("ERROR: Missing Dependencies")
#         print_error(f"Required packages not found: {', '.join(missing)}")
#         print()
#         print("Install with:")
#         print(f"  pip install {' '.join(missing)}")
#         print()
#         print("Or reinstall all requirements:")
#         print("  pip install -r python/requirements.txt")
#         print()
#         sys.exit(1)

# def check_pytorch():
#     """Check if PyTorch is installed."""
#     try:
#         import torch
#         import torchvision
#         print_success(f"PyTorch {torch.__version__} detected")
#         print_info(f"Torchvision {torchvision.__version__}")
#         return True
#     except ImportError:
#         print_error("PyTorch not installed")
#         print()
#         print("PyTorch is REQUIRED to download and convert pretrained models.")
#         print()
#         print("Install PyTorch:")
#         print()
#         print("  For CPU only (recommended, ~800MB):")
#         print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
#         print()
#         print("  For GPU CUDA 11.8 (~2GB):")
#         print("    pip install torch torchvision")
#         print()
#         print("  For GPU CUDA 12.1+ (~2GB):")
#         print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
#         print()
#         return False

# def create_simple_model():
#     """Create a simple linear model: y = 2*x + 1"""
#     import onnx
#     from onnx import helper, TensorProto
    
#     input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 5])
#     output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 5])
    
#     scale = np.array([2.0], dtype=np.float32)
#     scale_tensor = helper.make_tensor('scale', TensorProto.FLOAT, [1], scale.tolist())
    
#     bias = np.array([1.0], dtype=np.float32)
#     bias_tensor = helper.make_tensor('bias', TensorProto.FLOAT, [1], bias.tolist())
    
#     mul_node = helper.make_node('Mul', inputs=['input', 'scale'], outputs=['scaled_output'])
#     add_node = helper.make_node('Add', inputs=['scaled_output', 'bias'], outputs=['output'])
    
#     graph = helper.make_graph(
#         [mul_node, add_node],
#         'simple_linear_model',
#         [input_tensor],
#         [output_tensor],
#         [scale_tensor, bias_tensor]
#     )
    
#     model = helper.make_model(graph, producer_name='miia')
#     model.opset_import[0].version = 13
#     onnx.checker.check_model(model)
    
#     return model

# def create_classification_model():
#     """Create a simple classification model with softmax"""
#     import onnx
#     from onnx import helper, TensorProto
    
#     input_tensor = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
#     output_tensor = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])
    
#     weights = np.random.randn(4, 3).astype(np.float32) * 0.1
#     weights_tensor = helper.make_tensor('weights', TensorProto.FLOAT, [4, 3], weights.flatten().tolist())
    
#     bias = np.zeros(3, dtype=np.float32)
#     bias_tensor = helper.make_tensor('bias', TensorProto.FLOAT, [3], bias.tolist())
    
#     matmul_node = helper.make_node('MatMul', inputs=['input', 'weights'], outputs=['matmul_output'])
#     add_node = helper.make_node('Add', inputs=['matmul_output', 'bias'], outputs=['logits'])
#     softmax_node = helper.make_node('Softmax', inputs=['logits'], outputs=['output'], axis=1)
    
#     graph = helper.make_graph(
#         [matmul_node, add_node, softmax_node],
#         'simple_classifier',
#         [input_tensor],
#         [output_tensor],
#         [weights_tensor, bias_tensor]
#     )
    
#     model = helper.make_model(graph, producer_name='miia')
#     model.opset_import[0].version = 13
#     onnx.checker.check_model(model)
    
#     return model

# def consolidate_external_data(model_path):
#     """Consolidate external data files into single .onnx file if they exist.
    
#     PyTorch's torch.onnx.export() doesn't have a parameter to prevent external
#     data creation. It may create .data files automatically for large models.
#     This function consolidates them back into a single file.
#     """
#     import onnx
    
#     data_file = model_path + '.data'
    
#     # Check if .data file exists
#     if not os.path.exists(data_file):
#         return  # No external data, nothing to do
    
#     print_info(f"Found external data file: {os.path.basename(data_file)}")
#     print_info("Consolidating into single file...")
    
#     try:
#         # Load model with external data (reads both .onnx and .data)
#         model = onnx.load(model_path, load_external_data=True)
        
#         # Save without external data (everything embedded in .onnx)
#         onnx.save(model, model_path, save_as_external_data=False)
        
#         # Remove the .data file (no longer needed)
#         os.remove(data_file)
        
#         # Get new size
#         file_size = os.path.getsize(model_path) / (1024 * 1024)
#         print_success(f"Consolidated into single file ({file_size:.1f} MB)")
        
#     except Exception as e:
#         print_warning(f"Could not consolidate: {e}")
#         print_info("Model will use separate .data file")
#         print_info("Both .onnx and .data files are required!")

# def convert_pytorch_to_onnx(model_name, output_path, input_shape=(1, 3, 224, 224)):
#     """Download PyTorch model and convert to ONNX."""
#     import torch
#     import torchvision.models as models
#     import onnx
    
#     print_info(f"Downloading pretrained weights from PyTorch...")
    
#     try:
#         # Load model with pretrained weights
#         if model_name == 'resnet18':
#             model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
#         elif model_name == 'resnet50':
#             model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
#         elif model_name == 'mobilenet_v2':
#             model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
#         elif model_name == 'squeezenet1_0':
#             model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
#         elif model_name == 'efficientnet_b0':
#             model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
#         else:
#             print_error(f"Unknown model: {model_name}")
#             return False
        
#         print_success("Pretrained weights downloaded")
#         model.eval()
        
#         print_info(f"Converting to ONNX format...")
        
#         # Create dummy input
#         dummy_input = torch.randn(*input_shape)
        
#         # Export to ONNX
#         torch.onnx.export(
#             model,
#             dummy_input,
#             output_path,
#             input_names=['input'],
#             output_names=['output'],
#             dynamic_axes={
#                 'input': {0: 'batch_size'},
#                 'output': {0: 'batch_size'}
#             },
#             opset_version=13,
#             do_constant_folding=True,
#             export_params=True
#         )
        
#         print_success("Converted to ONNX")
        
#         # Validate and ensure single file
#         print_info("Validating ONNX model...")
#         onnx_model = onnx.load(output_path)
#         onnx.checker.check_model(onnx_model)
        
#         file_size = os.path.getsize(output_path) / (1024 * 1024)
#         print_success(f"Model validated successfully ({file_size:.1f} MB)")
#         print_info(f"Saved to: {output_path}")
        
#         # Consolidate external data if any was created (failsafe)
#         consolidate_external_data(output_path)
        
#         return True
        
#     except Exception as e:
#         print_error(f"Failed: {e}")
#         print()
#         print_info("Common issues:")
#         print_info("  - Missing onnxscript: pip install onnxscript")
#         print_info("  - Old PyTorch version: pip install --upgrade torch torchvision")
#         print_info("  - Memory issues: Try a smaller model first")
#         print()
#         return False

# def main():
#     check_dependencies()
    
#     import onnx
    
#     print_section("ML Inference - ONNX Model Creation")
    
#     # Check PyTorch
#     print("Checking PyTorch installation...")
#     pytorch_available = check_pytorch()
#     print()
    
#     if not pytorch_available:
#         print_warning("PyTorch is required for pretrained models.")
#         print_warning("Only simple test models will be created.")
#         print()
#         response = input("Continue anyway? (y/N): ").lower().strip()
#         if response != 'y':
#             print("\nInstall PyTorch and run again:")
#             print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
#             sys.exit(1)
    
#     # Create output directory
#     output_dir = 'models'
#     os.makedirs(output_dir, exist_ok=True)
    
#     print_section("Creating ONNX Models")
    
#     success_count = 0
#     total_count = 0
    
#     # Simple test models
#     print(f"{Colors.YELLOW}1. Simple Linear Model (y = 2*x + 1){Colors.NC}")
#     try:
#         linear_model = create_simple_model()
#         linear_path = os.path.join(output_dir, 'simple_linear.onnx')
#         onnx.save(linear_model, linear_path)
        
#         file_size = os.path.getsize(linear_path) / 1024
#         print_success(f"Created successfully ({file_size:.1f} KB)")
#         print_info(f"Input:  'input' [1, 5] float32")
#         print_info(f"Output: 'output' [1, 5] float32")
#         success_count += 1
#     except Exception as e:
#         print_error(f"Failed: {e}")
#     total_count += 1
    
#     print(f"\n{Colors.YELLOW}2. Simple Classification Model{Colors.NC}")
#     try:
#         classifier_model = create_classification_model()
#         classifier_path = os.path.join(output_dir, 'simple_classifier.onnx')
#         onnx.save(classifier_model, classifier_path)
        
#         file_size = os.path.getsize(classifier_path) / 1024
#         print_success(f"Created successfully ({file_size:.1f} KB)")
#         print_info(f"Input:  'input' [1, 4] float32")
#         print_info(f"Output: 'output' [1, 3] float32")
#         success_count += 1
#     except Exception as e:
#         print_error(f"Failed: {e}")
#     total_count += 1
    
#     # Pretrained models
#     if pytorch_available:
#         print_section("Converting Pretrained Models from PyTorch")
        
#         models_to_create = [
#             {
#                 'name': 'resnet18',
#                 'filename': 'resnet18.onnx',
#                 'label': '3. ResNet-18',
#                 'description': 'ImageNet classifier (~44 MB)'
#             },
#             {
#                 'name': 'mobilenet_v2',
#                 'filename': 'mobilenet_v2.onnx',
#                 'label': '4. MobileNet V2',
#                 'description': 'Lightweight classifier (~14 MB)'
#             },
#             {
#                 'name': 'squeezenet1_0',
#                 'filename': 'squeezenet1_0.onnx',
#                 'label': '5. SqueezeNet 1.0',
#                 'description': 'Very lightweight classifier (~5 MB)'
#             }
#         ]
        
#         for model_info in models_to_create:
#             print(f"\n{Colors.YELLOW}{model_info['label']}{Colors.NC}")
#             print_info(f"Description: {model_info['description']}")
#             print_info(f"Input:  [1, 3, 224, 224] (RGB image)")
#             print_info(f"Output: [1, 1000] (ImageNet classes)")
            
#             output_path = os.path.join(output_dir, model_info['filename'])
            
#             if convert_pytorch_to_onnx(model_info['name'], output_path):
#                 success_count += 1
            
#             total_count += 1
    
#     # Summary
#     print_section("Summary")
    
#     print(f"Models created: {Colors.GREEN}{success_count}/{total_count}{Colors.NC}\n")
    
#     # List models
#     print("Available models:")
#     if os.path.exists(output_dir):
#         model_files = sorted([f for f in os.listdir(output_dir) if f.endswith('.onnx')])
#         if model_files:
#             for filename in model_files:
#                 filepath = os.path.join(output_dir, filename)
#                 file_size = os.path.getsize(filepath) / (1024 * 1024)
#                 size_str = f"{file_size:.1f} MB" if file_size >= 1 else f"{file_size * 1024:.1f} KB"
#                 print(f"  {Colors.GREEN}✓{Colors.NC} {filename:<30} ({size_str})")
#         else:
#             print(f"  {Colors.YELLOW}No models found{Colors.NC}")
    
#     # Usage
#     print_section("Usage Instructions")
    
#     print("Test the models:")
#     print()
#     print(f"  {Colors.BLUE}# Terminal 1: Start worker{Colors.NC}")
#     print(f"  make run-worker")
#     print()
#     print(f"  {Colors.BLUE}# Terminal 2: Run client{Colors.NC}")
#     print(f"  make run-client")
#     print()
    
#     if success_count == total_count:
#         print(f"{Colors.GREEN}{'='*60}{Colors.NC}")
#         print(f"{Colors.GREEN}✓ All models created successfully!{Colors.NC}")
#         print(f"{Colors.GREEN}{'='*60}{Colors.NC}")
#     else:
#         print(f"{Colors.YELLOW}{'='*60}{Colors.NC}")
#         print(f"{Colors.YELLOW}⚠ {total_count - success_count} model(s) failed{Colors.NC}")
#         print(f"{Colors.YELLOW}{'='*60}{Colors.NC}")

# if __name__ == '__main__':
#     main()