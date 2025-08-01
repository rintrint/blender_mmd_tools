# Copyright 2014 MMD Tools authors
# This file is part of MMD Tools.

from bpy.props import BoolProperty, EnumProperty, FloatProperty
from bpy.types import Operator

from ..bpyutils import FnContext
from ..core.camera import MMDCamera


class ConvertToMMDCamera(Operator):
    bl_idname = "mmd_tools.convert_to_mmd_camera"
    bl_label = "Convert to MMD Camera"
    bl_description = "Create a camera rig for MMD"
    bl_options = {"REGISTER", "UNDO"}

    scale: FloatProperty(
        name="Scale",
        description="Scaling factor for initializing the camera",
        default=0.08,
    )

    bake_animation: BoolProperty(
        name="Bake Animation",
        description="Bake camera animation to a new MMD camera rig",
        default=False,
        options={"SKIP_SAVE"},
    )

    camera_source: EnumProperty(
        name="Camera Source",
        description="Select camera source to bake animation (camera target is the selected or DoF object)",
        items=[
            ("CURRENT", "Current", "Current active camera object", 0),
            ("SCENE", "Scene", "Scene camera object", 1),
        ],
        default="CURRENT",
    )

    min_distance: FloatProperty(
        name="Min Distance",
        description="Minimum distance to camera target when baking animation",
        default=0.1,
    )

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == "CAMERA"

    def invoke(self, context, event):
        vm = context.window_manager
        return vm.invoke_props_dialog(self)

    def execute(self, context):
        if self.bake_animation:
            obj = context.active_object
            targets = [x for x in context.selected_objects if x != obj]
            target = targets[0] if len(targets) == 1 else None
            if self.camera_source == "SCENE":
                obj = None
            camera = MMDCamera.newMMDCameraAnimation(obj, target, self.scale, self.min_distance).camera()
            FnContext.set_active_object(context, camera)
        else:
            MMDCamera.convertToMMDCamera(context.active_object, self.scale)
        return {"FINISHED"}
