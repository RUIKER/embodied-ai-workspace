from setuptools import setup


setup(
    name="panda_sim",
    version="0.0.1",
    packages=["panda_sim"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/panda_sim"]),
        ("share/panda_sim", ["package.xml"]),
        ("share/panda_sim/launch", ["launch/panda_demo.launch.py"]),
    ],
    install_requires=["setuptools", "mujoco", "numpy"],
    entry_points={"console_scripts": ["panda_sim_node = panda_sim.node:main"]},
)
