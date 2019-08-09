# Java Time Series Classification code to use in R 
This repo is the java backend of our R package for Time Series Classification: https://github.com/compstat-lmu/TSClassification

We adapt the following repository with Author's permission: https://github.com/TonyBagnall/uea-tsc.git

# Intellij: 
- make sure when project is imported, all dependencies are there, so always import from clean source recursively without iml file
- rm -rf .idea
- rm TimeSeriesClassification.iml
- rm TimeSeriesClassification/subfolder/.iml
- delete TimeSeriesClassification/src/META-INF/  inside Intellij
- Project Structure/Setting/artifact/jar(from modules and dependencies) (choose main class MainEntrance(must be searchable when click on the right folder icon instead of typing the name)) ---> click build/build-artifact, then jar will be generated(simply click build would not generate the jar)
- in order to depend on the up to date code base, you need to clone the most recent source code from https://github.com/TonyBagnall/uea-tsc.git and replace the folder in tsc4R/tsc4r/uea-tsc.git, then the tsc4r project will import this as dependency. 
