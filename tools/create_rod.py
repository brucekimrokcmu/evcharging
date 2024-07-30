import FreeCAD as App
import Part
import Mesh
import os

print("Starting the script execution...")

try:
    radius = 0.02
    length = 0.3
    filename = "rod.obj"

    print("Creating a new document...")
    doc = App.newDocument()

    print("Creating a cylinder...")
    cylinder = Part.makeCylinder(radius, length)

    print("Creating a shape from the cylinder...")
    shape = Part.Shape(cylinder)

    print("Creating a mesh from the shape...")
    mesh = Mesh.Mesh(shape.tessellate(0.01))  # 0.01 is the mesh deviation

    # Define the output path
    output_path = os.path.join("/home/brucekimrok/RoboticsProjects/evcharging_ws/data/universal_robots_ur10e/assets", filename)

    print(f"Ensuring the directory {os.path.dirname(output_path)} exists...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Exporting the mesh as OBJ to {output_path}...")
    mesh.write(output_path)

    print(f"Mesh saved as {output_path}")

    print("Closing the document without saving...")
    App.closeDocument(doc.Name)

except Exception as e:
    print(f"An error occurred: {e}")

