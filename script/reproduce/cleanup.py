with open("environment_full.yml") as f:
    lines = f.readlines()

with open("environment.yml", "w") as out:
    for line in lines:
        if "@" not in line and "file://" not in line:
            out.write(line)