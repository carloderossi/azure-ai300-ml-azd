import yaml
with open("src/data/test/MLTable", "r", encoding="utf8") as f:
    print(yaml.safe_load(f))