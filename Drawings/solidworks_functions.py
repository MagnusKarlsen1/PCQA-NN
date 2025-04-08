import win32com.client
import pythoncom
from win32com.client import VARIANT
import os



def start_SW():
    swApp = win32com.client.Dispatch("SldWorks.Application")
    swApp.Visible = True

    return swApp


def open_part(swApp, part_path):
    part_path = os.path.abspath(part_path)

    if not os.path.exists(part_path):
        raise FileNotFoundError(f"Part file does not exist: {part_path}")

    print(f"✅ Trying to open part: {part_path}")

    errors = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    warnings = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)

    model = swApp.OpenDoc6(part_path, 1, 0, "", errors, warnings)

    if model is None:
        raise RuntimeError("❌ SolidWorks failed to open the part file. Check file format and licensing.")

    return model



def save_as_stl(model, output_path):
    output_path = os.path.abspath(output_path)
    print("💾 Saving to:", output_path)

    model.SaveAs3(output_path, 0, 2)
    


def rebuild_model(model):
    model.ForceRebuild3(False)


def close_part(swApp, model):
    swApp.CloseDoc(model.GetTitle())
    
    
def update_parameter_file(param_file, params):
    with open(param_file, 'w') as f:
        for name, value in params.items():
            f.write(f"{name}={value}\n")

#######################################################################

def get_part_and_params(shape="ball"):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    if shape == "ball": 
        part_path = os.path.join(BASE_DIR, "Ball", "Ball.SLDPRT")
        params_file = os.path.join(BASE_DIR, "Ball", "parameters.txt")

    elif shape == "saddle":
        part_path = os.path.join(BASE_DIR, "Saddle", "Saddle.SLDPRT")
        params_file = os.path.join(BASE_DIR, "Saddle", "parameters.txt")

    elif shape == "box":
        part_path = os.path.join(BASE_DIR, "Edge", "Box.SLDPRT")
        params_file = os.path.join(BASE_DIR, "Edge", "parameters.txt")

    elif shape == "curve":
        part_path = os.path.join(BASE_DIR, "Curved surface", "Curved_surface.SLDPRT")
        params_file = os.path.join(BASE_DIR, "Curved surface", "parameters.txt")
    elif shape == "wedge":
        part_path = os.path.join(BASE_DIR, "wedge", "wedge.SLDPRT")
        params_file = os.path.join(BASE_DIR, "wedge", "parameters.txt")
    elif shape == "donut":
        part_path = os.path.join(BASE_DIR, "Donut", "donut.SLDPRT")
        params_file = os.path.join(BASE_DIR, "Donut", "parameters.txt")
    elif shape == "donut_center":
        part_path = os.path.join(BASE_DIR, "Donut_w_center", "Donut_w_center.SLDPRT")
        params_file = os.path.join(BASE_DIR, "Donut_w_center", "parameters.txt")
    elif shape == "nut":
        part_path = os.path.join(BASE_DIR, "Nut", "nut.SLDPRT")
        params_file = os.path.join(BASE_DIR, "Nut", "parameters.txt")
    elif shape == "heart":
        part_path = os.path.join(BASE_DIR, "Heart", "heart.SLDPRT")
        params_file = os.path.join(BASE_DIR, "Heart", "parameters.txt")
    elif shape == "mix":
        part_path = [
            os.path.join(BASE_DIR, "Ball", "Ball.SLDPRT"),
            os.path.join(BASE_DIR, "Saddle", "Saddle.SLDPRT"),
            os.path.join(BASE_DIR, "Edge", "Box.SLDPRT"),
            os.path.join(BASE_DIR, "Curved surface", "Curved_surface.SLDPRT")
        ]
        params_file = [
            os.path.join(BASE_DIR, "Ball", "parameters.txt"),
            os.path.join(BASE_DIR, "Saddle", "parameters.txt"),
            os.path.join(BASE_DIR, "Edge", "parameters.txt"),
            os.path.join(BASE_DIR, "Curved surface", "parameters.txt")
        ]
    
    else:
        raise RuntimeError("❌ Missing or wrong shape input parameter!")

    return part_path, params_file

def Create_geometry(shape: str, output_path: str, params: dict):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    
    if shape == "ball":
        part_path = os.path.join(BASE_DIR, "Ball", "Ball.SLDPRT")
        params_file = os.path.join(BASE_DIR, "Ball", "parameters.txt")
    elif shape == "saddle":
        part_path = os.path.join(BASE_DIR, "Saddle", "Saddle.SLDPRT")
        params_file = os.path.join(BASE_DIR, "Saddle", "parameters.txt")
    elif shape == "box":
        part_path = os.path.join(BASE_DIR, "Edge", "Box.SLDPRT")
        params_file = os.path.join(BASE_DIR, "Edge", "parameters.txt")
    elif shape == "curve":
        part_path = os.path.join(BASE_DIR, "Curved surface", "Curved_surface.SLDPRT")
        params_file = os.path.join(BASE_DIR, "Curved surface", "parameters.txt")
    elif shape == "angle_curve":
        part_path = os.path.join(BASE_DIR, "Angle_curve", "angle_curve.SLDPRT")
        params_file = os.path.join(BASE_DIR, "Angle_curve", "equations.txt")
    elif shape == "wedge":
        part_path = os.path.join(BASE_DIR, "wedge", "wedge.SLDPRT")
        params_file = os.path.join(BASE_DIR, "wedge", "parameters.txt")
    elif shape == "donut":
        part_path = os.path.join(BASE_DIR, "Donut", "donut.SLDPRT")
        params_file = os.path.join(BASE_DIR, "Donut", "parameters.txt")
    elif shape == "donut_center":
        part_path = os.path.join(BASE_DIR, "Donut_w_center", "Donut_w_center.SLDPRT")
        params_file = os.path.join(BASE_DIR, "Donut_w_center", "parameters.txt")
    elif shape == "nut":
        part_path = os.path.join(BASE_DIR, "Nut", "nut.SLDPRT")
        params_file = os.path.join(BASE_DIR, "Nut", "parameters.txt")
    elif shape == "heart":
        part_path = os.path.join(BASE_DIR, "Heart", "heart.SLDPRT")
        params_file = os.path.join(BASE_DIR, "Heart", "parameters.txt")
    
    else:
        raise RuntimeError("❌ Missing or wrong shape input parameter!")
    
    errors = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    warnings = VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0)
    
    swApp = start_SW()
    
    model = open_part(swApp, part_path)
    swApp.ActivateDoc3(os.path.basename(part_path), 1, errors, warnings)
    model = swApp.ActiveDoc

    # sf.rebuild_model(model)
    update_parameter_file(params_file, params)
    rebuild_model(model)
    save_as_stl(model, output_path)
    
    print(f"Geometry created and saved to {output_path}")
    return model
    
def delete_stl(path):
    if os.path.exists(path):
        os.remove(path)
    else:
        print("The file does not exist") 



def get_surface_area(model, unit = "mm2"):
    """
    Uses GetMassProperties2 to get surface area from the full model.
    Finds the Surface area and gives it in mm
    """

    mass_props = model.Extension.GetMassProperties2(1, VARIANT(pythoncom.VT_BYREF | pythoncom.VT_I4, 0), False)

    if not mass_props:
        raise RuntimeError("❌ GetMassProperties2 failed.")

    if unit == "mm2":
        
        print(f"Surface Area: {mass_props[4]*1_000_000} mm^2")
        return mass_props[4]*1_000_000
    elif unit == "m2":
        print(f"Surface Area: {mass_props[4]} m^2")
        return mass_props[4]
    else:
        return print("wrong unit, use 'mm2' or 'm2'")
