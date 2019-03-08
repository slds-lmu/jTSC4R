# Refactoring Time Series Classification for R 
https://github.com/compstat-lmu/TSClassification

# Intellij: make sure when project is imported, all dependencies are there, so always import from clean source recursively without iml file
- rm -rf .idea
- rm TimeSeriesClassification.iml
- rm TimeSeriesClassification/subfolder/.iml
- delete TimeSeriesClassification/src/META-INF/  inside Intellij
- Project Structure/Setting/artifact/jar (choose main class MainEntrance(must be searchable)) ---> click build

