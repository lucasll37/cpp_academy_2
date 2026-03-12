from conan import ConanFile
from conan.tools.gnu import PkgConfigDeps
from conan.tools.meson import MesonToolchain, Meson


class MLInferenceRecipe(ConanFile):
    name = "asa-poc-miia"
    version = "1.0.0"

    license = "MIT"
    url = "https://github.com/lucasll37/poc-miia"
    description = "ML Inference System with gRPC and ONNX Runtime"
    settings = "arch", "build_type", "compiler", "os"
    
    options = {"enable_gpu": [True, False]}
    default_options = {"enable_gpu": False}
    
    exports_sources = "meson*", "cpp/*", "scripts/*", "proto/*", "conanfile.py", "asa-poc-miia.pc.in"
    
    # def build_requirements(self):
    #     self.tool_requires("cmake/3.31.9")  
    #     self.tool_requires("meson/2.24.0")
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
        self.cpp_info.libs = ["asa_miia_client"]
        self.cpp_info.includedirs = ["include"]