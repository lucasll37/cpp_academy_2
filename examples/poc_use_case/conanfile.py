from subprocess import PIPE, Popen

from conan import ConanFile
from conan.tools.meson import Meson, MesonToolchain
from conan.tools.gnu import PkgConfigDeps


class Recipe(ConanFile):
    name = "asa-tutorial"

    url = "http://gitlab.asa.dcta.mil.br/asa/asa-tutorial"
    license = "Copyright © 2021-2023 ASA. All rights reserved."
    description = "Asa tutorial model."

    settings = "os", "arch", "compiler", "build_type"
    options = {"fPIC": [True, False]}
    default_options = {"fPIC": True}

    exports_sources = (
        "include/*",
        "resources/*",
        "src/*",
        "asa-tutorial.pc.in",
        "meson.build",
        "version.sh",
    )

    @property
    def version(self) -> str:
        if ConanFile.version is None:
            proc = Popen("./version.sh", stdout=PIPE)
            out = proc.stdout.read()
            return out.decode()
        else:
            return ConanFile.version

    @version.setter
    def version(self, value):
        ConanFile.version = value

    def requirements(self):
        # 2nd party
        self.requires("asa-models/1.0.5")
        self.requires("asa-poc-miia/1.0.0")
        self.requires("gtest/1.14.0")
        # 3rd party
        self.requires("rapidjson/cci.20230929")

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def layout(self):
        self.folders.build = "build"
        self.folders.generators = "build"

    def generate(self):
        tc = PkgConfigDeps(self)
        tc.generate()

        tc = MesonToolchain(self)
        tc.generate()

    def build(self):
        meson = Meson(self)
        meson.configure()
        meson.build()

    def package(self):
        meson = Meson(self)
        meson.install()

    def package_info(self):
        self.cpp_info.components["tutorial"].libs = ["asa_tutorial"]
