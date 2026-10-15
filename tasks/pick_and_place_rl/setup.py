from setuptools import find_packages, setup

package_name = "pick_and_place_rl"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Dylan K",
    maintainer_email="kainth.dylan@gmail.com",
    description="Learned TIAGo tabletop pick and place in Gazebo",
    license="Apache-2.0",
    entry_points={"console_scripts": []},
)
