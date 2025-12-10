from setuptools import find_packages, setup


setup(
	name="textsentimentanalysis-ai",
	version="0.1.0",
	description="Text sentiment analysis pipeline with baseline models",
	author="Matheus Braum",
	packages=find_packages(where="src"),
	package_dir={"": "src"},
	install_requires=[
		"pandas>=2.3.3,<3.0.0",
		"scikit-learn>=1.7.2,<2.0.0",
		"joblib>=1.5.2,<2.0.0",
		"matplotlib>=3.8,<4.0",
		"seaborn>=0.13,<1.0",
		"numpy>=1.26,<2.0",
	],
)
