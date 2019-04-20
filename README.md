# Time Series Classification in R 
This repo is the java backend of our R package for Time Series Classification: https://github.com/compstat-lmu/TSClassification

We adapt the following repository with Author's permission: https://bitbucket.org/TonyBagnall/time-series-classification/src/default/

# Intellij: 
- make sure when project is imported, all dependencies are there, so always import from clean source recursively without iml file
- rm -rf .idea
- rm TimeSeriesClassification.iml
- rm TimeSeriesClassification/subfolder/.iml
- delete TimeSeriesClassification/src/META-INF/  inside Intellij
- Project Structure/Setting/artifact/jar(from modules and dependencies) (choose main class MainEntrance(must be searchable when click on the right folder icon instead of typing the name)) ---> click build/build-artifact, then jar will be generated(simply click build would not generate the jar)
