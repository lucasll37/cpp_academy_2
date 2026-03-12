from conan import ConanFile
from conan.tools.gnu import PkgConfigDeps
from conan.tools.meson import MesonToolchain, Meson


class MLInferenceRecipe(ConanFile):
    name = "apaga"
    version = "1.0.0"

    license = "MIT"
    url = "https://github.com/lucasll37/poc-miia"
    description = "ML Inference System with gRPC and ONNX Runtime"
    settings = "arch", "build_type", "compiler", "os"
    
    exports_sources = "meson*", "include/*", "src/*", "conanfile.py"
    
    # def build_requirements(self):
    #     self.tool_requires("meson")
    #     self.tool_requires("pkgconf")
    #     self.tool_requires("ninja")

    def requirements(self):
        self.requires("mixr/1.0.5")
        self.requires("asa-poc-miia/1.0.0")

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
        self.cpp_info.includedirs = ["include"]