var = "lorem\\ipsum"
print(f"Variable: {var}") #String Display
print(f"Variable: {var!r} (string representation)") #String Representation
print(f"Variable: {var:{20}} (with minimum spacing)") #Minimum Width Limit
print(f"Variable: {var:-<{20}} {var:^{20}} {var:^{20}} {var:.>{20}} (with minimum spacing formatted)") #Minimum With Limit Formatted
