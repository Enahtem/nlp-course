
# Writing to File (Creates/Overrides)
with open("text.txt", "w") as file:
    file.write("Line 1.\nLine 2.")

# Appending to File
with open("text.txt", "a") as file:
    file.write("\nLine 3 (Appended)")

# Reading from File
with open("text.txt", "r") as file:
    content=file.read()
    print(content)


# Code for File location
"""
import os
file_path = os.path.abspath("text.txt")
print("File Location:", file_path)
"""
