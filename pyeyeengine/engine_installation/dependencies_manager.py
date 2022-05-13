import apt
import pkg_resources
from pyeyeengine.engine_installation import install_script_generator as Gen


def check_dependencies():
    satisfied_dependencies = True
    missing_packages = {}
    wrong_versions = list()

    ## pip3 ##
    pip3_installed = {pkg.key for pkg in pkg_resources.working_set}
    pip3_missing = list(Gen.pip3_required_packages.keys() - pip3_installed)

    for package in Gen.pip3_required_packages:
        if (package in pip3_missing) == False:
            required_version = Gen.pip3_required_packages[package]

            if required_version == "any":
                continue

            installed_version = pkg_resources.get_distribution(package).version

            if installed_version != required_version:
                pip3_missing.append("{}=={}".format(package, required_version))
                wrong_versions.append("{}-> installed: {} required: {}".format(package, installed_version, required_version))

    if len(wrong_versions) > 0:
        satisfied_dependencies = False
        missing_packages['version_mismatch'] = wrong_versions

    if len(pip3_missing) > 0:
        satisfied_dependencies = False
        missing_packages['pip3'] = pip3_missing

    ## apt ##
    cache = apt.Cache()
    apt_missing = list()

    for package in Gen.apt_required_packages:
        try:
            cached_package = cache[package]

            if cached_package.is_installed == False:
                apt_missing.append(package)
        except:
            apt_missing.append(package)

    if len(apt_missing) > 0:
        satisfied_dependencies = False
        missing_packages['apt'] = apt_missing

    description = ', '.join(str(missing) for missing in (list(pip3_missing) + list(apt_missing)))

    return satisfied_dependencies, missing_packages, description

class DependencyException(Exception):
    def __init__(self, message=""):
        super().__init__(message)