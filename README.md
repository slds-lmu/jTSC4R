# Refactoring Java Time Series Classification code for R 
- original code: https://bitbucket.org/TonyBagnall/time-series-classification/src/default/
- target R package: https://github.com/compstat-lmu/TSClassification

# Intellij: 
- make sure when project is imported, all dependencies are there, so always import from clean source recursively without iml file
- rm -rf .idea
- rm TimeSeriesClassification.iml
- rm TimeSeriesClassification/subfolder/.iml
- delete TimeSeriesClassification/src/META-INF/  inside Intellij
- Project Structure/Setting/artifact/jar(from modules and dependencies) (choose main class MainEntrance(must be searchable when click on the right folder icon instead of typing the name)) ---> click build/build-artifact, then jar will be generated(simply click build would not generate the jar)
