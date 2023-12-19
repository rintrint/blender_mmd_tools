import bpy

class PresetOptions(bpy.types.PropertyGroup):
    """
    Property group used during conversion of node group to python
    """
    dir_path : bpy.props.StringProperty(
        name = "Save Location",
        subtype='DIR_PATH',
        description="Save location if exporting preset",
        default = "//preset/"
    )
