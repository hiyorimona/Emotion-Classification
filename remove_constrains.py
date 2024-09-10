# poetry export -f requirements.txt --without-hashes --output requirements.txt
with open("requirements.txt", "r") as file:
    lines = file.readlines()

with open("requirements.txt", "w") as file:
    for line in lines:
        # Split the line at ';' to remove the python_version part if it exists
        clean_line = line.split(';')[0].strip()
        # Write the cleaned line to the file if it contains a package
        if clean_line:
            file.write(clean_line + '\n')