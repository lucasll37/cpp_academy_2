from conan import ConanFile
from conan.tools.gnu import PkgConfigDeps
from conan.tools.meson import MesonToolchain, Meson
from conan.tools.system.package_manager import Apt


class miiaRecipe(ConanFile):
    name = "miia"
    version = "1.0.0"

    license = "MIT"
    url = "https://gitlab.asa.dcta.mil.br/asa/miia.git"
    description = "ML Inference System with gRPC and ONNX Runtime"
    settings = "arch", "build_type", "compiler", "os"
    
    options = {"enable_gpu": [True, False], "enable_docs": [True, False], "build_tests": [True, False]}
    
    default_options = {
        "enable_gpu": False,
        "enable_docs": True,
        "build_tests": True
    }
    
    exports_sources = "meson*", "core/*", "tests/*", "proto/*", "docs/*", "python/meson.build", "conanfile.py", "miia.pc.in"
    
    
    def system_requirements(self):
        apt = Apt(self)
        # apt.update() # requer sudo
        apt.install(["python3.12", "python3.12-dev", "python3.12-venv"]) # "python3-numpy"
        apt.install(["lcov"])
        apt.install(["graphviz"])
        apt.install(["clang-format-18", "clang-tidy-18"])

    # def build_requirements(self): # requer source ./build/conanbuild.sh
    #     self.tool_requires("doxygen/1.9.4")
    #     self.tool_requires("cmake/3.24.4")  
    #     self.tool_requires("meson/1.3.2")
    #     self.tool_requires("ninja/1.11.1")
    
    def requirements(self):
        self.requires("grpc/1.54.3")
        self.requires("protobuf/3.21.12", override=True)
        self.requires("abseil/20230802.1", override=True)
        self.requires("re2/20230301", override=True)
        self.requires("onnxruntime/1.18.1")
        self.requires("gtest/1.14.0")

    def generate(self):
        pc = PkgConfigDeps(self)
        pc.generate()
        
        tc = MesonToolchain(self)
        tc.generate()
        
    def layout(self):
        self.folders.build = "build"
        self.folders.generators = "build"
        
    def build(self):
        meson = Meson(self)
        meson.configure()
        meson.build()

    def package(self):
        meson = Meson(self)
        meson.install()

    def package_info(self):
        self.cpp_info.libs = ["miia_client"]
        self.cpp_info.includedirs = ["include"]