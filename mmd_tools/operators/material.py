# -*- coding: utf-8 -*-

import traceback
import os
from .material_utils import *
from io import StringIO
import mathutils

import bpy
from bpy.types import Operator
from bpy.types import OperatorFileListElement
from bpy.props import StringProperty, BoolProperty
from bpy_extras.io_utils import ImportHelper, ExportHelper

from mmd_tools import cycles_converter
from mmd_tools.core.material import FnMaterial
from mmd_tools.core.exceptions import MaterialNotFoundError
from mmd_tools.core.shader import _NodeGroupUtils

class ConvertMaterialsForCycles(Operator):
    bl_idname = 'mmd_tools.convert_materials_for_cycles'
    bl_label = 'Convert Materials For Cycles'
    bl_description = 'Convert materials of selected objects for Cycles.'
    bl_options = {'REGISTER', 'UNDO'}

    use_principled: bpy.props.BoolProperty(
        name='Convert to Principled BSDF',
        description='Convert MMD shader nodes to Principled BSDF as well if enabled',
        default=False,
        options={'SKIP_SAVE'},
        )

    clean_nodes: bpy.props.BoolProperty(
        name='Clean Nodes',
        description='Remove redundant nodes as well if enabled. Disable it to keep node data.',
        default=False,
        options={'SKIP_SAVE'},
        )

    @classmethod
    def poll(cls, context):
        return next((x for x in context.selected_objects if x.type == 'MESH'), None)

    def draw(self, context):
        layout = self.layout
        if cycles_converter.is_principled_bsdf_supported():
            layout.prop(self, 'use_principled')
        layout.prop(self, 'clean_nodes')

    def execute(self, context):
        try:
            context.scene.render.engine = 'CYCLES'
        except:
            self.report({'ERROR'}, ' * Failed to change to Cycles render engine.')
            return {'CANCELLED'}
        for obj in (x for x in context.selected_objects if x.type == 'MESH'):
            cycles_converter.convertToCyclesShader(obj, use_principled=self.use_principled, clean_nodes=self.clean_nodes)
        return {'FINISHED'}

class ConvertMaterials(Operator):
    bl_idname = 'mmd_tools.convert_materials'
    bl_label = 'Convert Materials'
    bl_description = 'Convert materials of selected objects.'
    bl_options = {'REGISTER', 'UNDO'}

    use_principled: bpy.props.BoolProperty(
        name='Convert to Principled BSDF',
        description='Convert MMD shader nodes to Principled BSDF as well if enabled',
        default=True,
        options={'SKIP_SAVE'},
        )

    clean_nodes: bpy.props.BoolProperty(
        name='Clean Nodes',
        description='Remove redundant nodes as well if enabled. Disable it to keep node data.',
        default=True,
        options={'SKIP_SAVE'},
        )

    subsurface: bpy.props.FloatProperty(
        name='Subsurface',
        default=0.001,
        soft_min=0.000, soft_max=1.000,
        precision=3,
        options={'SKIP_SAVE'},
        )

    @classmethod
    def poll(cls, context):
        return next((x for x in context.selected_objects if x.type == 'MESH'), None)

    def execute(self, context):
        for obj in context.selected_objects:
            if obj.type != 'MESH':
                continue
            cycles_converter.convertToBlenderShader(obj, use_principled=self.use_principled, clean_nodes=self.clean_nodes, subsurface=self.subsurface)
        return {'FINISHED'}

class ImportPreset(Operator, ImportHelper):
    bl_idname = 'mmd_tools.import_material_preset'
    bl_label = 'Import Preset'
    bl_description = 'Import a material preset to selected objects.'
    bl_options = {'REGISTER', 'UNDO'}
    
    # Todo::不知道material_utils放哪好
    
    filepath: bpy.props.StringProperty(subtype="FILE_PATH")
    
    filename_ext = '.py'
    filter_glob: bpy.props.StringProperty(default='*.py', options={'HIDDEN'})
    
    clean_nodes: bpy.props.BoolProperty(
        name='Clean Nodes',
        description='Remove redundant nodes as well if enabled. Disable it to keep node data.',
        default=False,
        options={'SKIP_SAVE'},
        )

    @classmethod
    def poll(cls, context):
        return next((x for x in context.selected_objects if x.type == 'MESH'), None)

    def execute(self, context):
        print ("Selected file:", self.filepath)
        try:
            for obj in context.selected_objects:
                if obj.type != 'MESH':
                    continue
                for i in obj.material_slots:
                    if not i.material:
                        continue
                    
                    nodes = i.material.node_tree.nodes
                    for node in nodes:
                        node.select = False
                    matout_nodes = [node for node in nodes if node.type == 'OUTPUT_MATERIAL']
                    old_matout_highest = mathutils.Vector((-999999.0, -999999.0))
                    for matout_node in matout_nodes:
                        if matout_node.location.y > old_matout_highest.y:
                            old_matout_highest = matout_node.location
                    
                    # 避免.001的name(強迫症)
                    if self.clean_nodes:
                        nodes = i.material.node_tree.nodes
                        for node in nodes:
                            if node.name not in ['mmd_tex_uv', 'mmd_shader', 'mmd_base_tex', 'mmd_toon_tex', 'mmd_sphere_tex']:
                                nodes.remove(node)
                    
                    # Import Preset
                    global mat
                    mat = i.material
                    exec(open(self.filepath).read(), globals())
                    
                    node_tree = i.material.node_tree
                    nodes = i.material.node_tree.nodes
                    old_nodes = [node for node in nodes if node.select == False]
                    added_nodes = [node for node in nodes if node.select == True]
                    added_mmd_tex_uv_nodes = [node for node in added_nodes if node.label == 'MMDTexUV']
                    added_mmd_shader_nodes = [node for node in added_nodes if node.label == 'MMDShaderDev']
                    added_mmd_base_tex_nodes = [node for node in added_nodes if node.label == 'Mmd Base Tex']
                    added_mmd_toon_tex_nodes = [node for node in added_nodes if node.label == 'Mmd Toon Tex']
                    added_mmd_sphere_tex_nodes = [node for node in added_nodes if node.label == 'Mmd Sphere Tex']
                    added_matout_nodes = [node for node in added_nodes if node.type == 'OUTPUT_MATERIAL']
                    mmd_tex_uv_node = None
                    mmd_shader_node = None
                    mmd_base_tex_node = None
                    mmd_toon_tex_node = None
                    mmd_sphere_tex_node = None
                    
                    if 'mmd_tex_uv' in nodes:
                        mmd_tex_uv_node = nodes['mmd_tex_uv']
                    if 'mmd_shader' in nodes:
                        mmd_shader_node = nodes['mmd_shader']
                    if 'mmd_base_tex' in nodes:
                        mmd_base_tex_node = nodes['mmd_base_tex']
                    else:
                        for n in old_nodes:
                            if n.label == 'Mmd Base Tex' and n.image is not None:
                                mmd_base_tex_node = n
                                break
                    if 'mmd_toon_tex' in nodes:
                        mmd_toon_tex_node = nodes['mmd_toon_tex']
                    else:
                        for n in old_nodes:
                            if n.label == 'Mmd Toon Tex' and n.image is not None:
                                mmd_toon_tex_node = n
                                break
                    if 'mmd_sphere_tex' in nodes:
                        mmd_sphere_tex_node = nodes['mmd_sphere_tex']
                    else:
                        for n in old_nodes:
                            if n.label == 'Mmd Sphere Tex' and n.image is not None:
                                mmd_sphere_tex_node = n
                                break
                    
                    if mmd_base_tex_node is not None:
                        for n in added_mmd_base_tex_nodes:
                            n.image = mmd_base_tex_node.image
                            n.show_texture = True
                    if mmd_toon_tex_node is not None:
                        for n in added_mmd_toon_tex_nodes:
                            n.image = mmd_toon_tex_node.image
                    if mmd_sphere_tex_node is not None:
                        for n in added_mmd_sphere_tex_nodes:
                            n.image = mmd_sphere_tex_node.image
                    if added_matout_nodes != []:
                        added_matout_nodes[0].is_active_output = True
                    
                    # 相容MMD腳本 改Alpha 改名
                    for n in added_mmd_shader_nodes:
                        if 'Alpha' in n.inputs and 'Alpha' in mmd_shader_node.inputs:
                            n.inputs['Alpha'].default_value = mmd_shader_node.inputs['Alpha'].default_value
                        if 'Toon Tex Fac' in n.inputs and 'Toon Tex Fac' in mmd_shader_node.inputs:
                            n.inputs['Toon Tex Fac'].default_value = mmd_shader_node.inputs['Toon Tex Fac'].default_value
                        if 'Sphere Tex Fac' in n.inputs and 'Sphere Tex Fac' in mmd_shader_node.inputs:
                            n.inputs['Sphere Tex Fac'].default_value = mmd_shader_node.inputs['Sphere Tex Fac'].default_value
                        if 'Sphere Mul/Add' in n.inputs and 'Sphere Mul/Add' in mmd_shader_node.inputs:
                            n.inputs['Sphere Mul/Add'].default_value = mmd_shader_node.inputs['Sphere Mul/Add'].default_value
                        if 'Base Tex' in n.inputs and 'Base Tex' in mmd_shader_node.inputs:
                            n.inputs['Base Tex'].default_value = mmd_shader_node.inputs['Base Tex'].default_value
                        if 'Toon Tex' in n.inputs and 'Toon Tex' in mmd_shader_node.inputs:
                            n.inputs['Toon Tex'].default_value = mmd_shader_node.inputs['Toon Tex'].default_value
                        if 'Sphere Tex' in n.inputs and 'Sphere Tex' in mmd_shader_node.inputs:
                            n.inputs['Sphere Tex'].default_value = mmd_shader_node.inputs['Sphere Tex'].default_value
                    def __switch_mmd_node_name(name, node, new_nodes):
                        if new_nodes != []:
                            new_node = new_nodes[0]
                        if node is None:
                            new_node.name = name
                        else:
                            tmp1 = node.name
                            tmp2 = new_node.name
                            node.name = tmp2
                            new_node.name = tmp1
                            node.name = tmp2
                    __switch_mmd_node_name('mmd_tex_uv', mmd_tex_uv_node, added_mmd_tex_uv_nodes)
                    __switch_mmd_node_name('mmd_shader', mmd_shader_node, added_mmd_shader_nodes)
                    __switch_mmd_node_name('mmd_base_tex', mmd_base_tex_node, added_mmd_base_tex_nodes)
                    __switch_mmd_node_name('mmd_toon_tex', mmd_toon_tex_node, added_mmd_toon_tex_nodes)
                    __switch_mmd_node_name('mmd_sphere_tex', mmd_sphere_tex_node, added_mmd_sphere_tex_nodes)

                    if self.clean_nodes:
                        for old_node in old_nodes:
                            nodes.remove(old_node)
                    else:
                        # align added nodes
                        added_matout_lowest = mathutils.Vector((999999.0, 999999.0))
                        for added_matout_node in added_matout_nodes:
                            if added_matout_node.location.y < added_matout_lowest.y:
                                added_matout_lowest = added_matout_node.location
                        location_shift = mathutils.Vector((0.0, 660.0))
                        location_shift += old_matout_highest - added_matout_lowest
                        print("location_shift", location_shift)
                        if abs(location_shift.y) < 500000:
                            for added_node in added_nodes:
                                added_node.location = added_node.location + location_shift
                    
                    # 清除多餘的toon和sphere節點
                    for n in added_nodes:
                        if mmd_toon_tex_node is None and n.label == 'Mmd Toon Tex':
                                nodes.remove(n)
                                continue # 避免node已經被remove掉了
                        if mmd_sphere_tex_node is None and n.label == 'Mmd Sphere Tex':
                                nodes.remove(n)
                    
        except Exception as e:
            err_msg = traceback.format_exc()
            self.report({'ERROR'}, err_msg)
        self.report({'INFO'}, f"ImportPreset: Import {self.filepath} OK")
        return {'FINISHED'}

class ExportPreset(Operator):
    bl_idname = 'mmd_tools.export_material_preset'
    bl_label = 'Export Preset'
    bl_description = 'Export a material preset to selected objects.'
    bl_options = {'REGISTER', 'UNDO'}
    
    @classmethod
    def poll(cls, context):
        return next((x for x in context.selected_objects if x.type == 'MESH'), None)
    
    def execute(self, context):
        try:
            for obj in context.selected_objects:
                if obj.type != 'MESH':
                    continue
                    
                # 處理label以便導入時正常工作
                for i in obj.material_slots:
                    if not i.material:
                        continue
                    nodes = i.material.node_tree.nodes
                    if 'mmd_tex_uv' in nodes:
                        nodes['mmd_tex_uv'].label = 'MMDTexUV'
                    if 'mmd_shader' in nodes:
                        nodes['mmd_shader'].label = 'MMDShaderDev'
                    if 'mmd_base_tex' in nodes:
                        nodes['mmd_base_tex'].label = 'Mmd Base Tex'
                    if 'mmd_toon_tex' in nodes:
                        nodes['mmd_toon_tex'].label = 'Mmd Toon Tex'
                    if 'mmd_sphere_tex' in nodes:
                        nodes['mmd_sphere_tex'].label = 'Mmd Sphere Tex'
                
                if True:
                    # export active material
                    if obj.active_material:
                        self.material_name = obj.active_material.name
                        self.mode = 'SCRIPT'
                        self._material_to_python(context)
                else:
                    # export all material
                    for i in obj.material_slots:
                        if not i.material:
                            continue
                        self.material_name = i.material.name
                        self.mode = 'SCRIPT'
                        self._material_to_python(context)
        except Exception as e:
            err_msg = traceback.format_exc()
            self.report({'ERROR'}, err_msg)
        return {'FINISHED'}

    def _material_to_python(self, context):
        node_settings = {
        #input
        "ShaderNodeAmbientOcclusion" : ["samples", "inside", "only_local"],
        "ShaderNodeAttribute" : ["attribute_type", "attribute_name"],
        "ShaderNodeBevel" : ["samples"],
        "ShaderNodeVertexColor" : ["layer_name"],
        "ShaderNodeTangent" : ["direction_type", "axis"],
        "ShaderNodeTexCoord" : ["object", "from_instancer"],
        "ShaderNodeUVMap" : ["from_instancer", "uv_map"],
        "ShaderNodeWireframe" : ["use_pixel_size"],

        #output
        "ShaderNodeOutputAOV" : ["name"],
        "ShaderNodeOutputMaterial" : ["target"],

        #shader
        "ShaderNodeBsdfGlass" : ["distribution"],
        "ShaderNodeBsdfGlossy" : ["distribution"],
        "ShaderNodeBsdfPrincipled" : ["distribution", "subsurface_method"],
        "ShaderNodeBsdfRefraction" : ["distribution"],
        "ShaderNodeSubsurfaceScattering" : ["falloff"],

        #texture
        "ShaderNodeTexBrick" : ["offset", "offset_frequency", "squash", "squash_frequency"],
        "ShaderNodeTexEnvironment" : ["interpolation", "projection", "image_user.frame_duration", "image_user.frame_start", "image_user.frame_offset", "image_user.use_cyclic", "image_user.use_auto_refresh"],
        "ShaderNodeTexGradient" : ["gradient_type"],
        "ShaderNodeTexIES" : ["mode"],
        "ShaderNodeTexImage" : ["interpolation", "projection", "projection_blend", 
                                "extension"],
        "ShaderNodeTexMagic" : ["turbulence_depth"],
        "ShaderNodeTexMusgrave" : ["musgrave_dimensions", "musgrave_type"],
        "ShaderNodeTexNoise" : ["noise_dimensions"],
        "ShaderNodeTexPointDensity" : ["point_source", "object", "space", "radius", 
                                        "interpolation", "resolution", 
                                        "vertex_color_source"],
        "ShaderNodeTexSky" : ["sky_type", "sun_direction", "turbidity",
                                "ground_albedo", "sun_disc", "sun_size", 
                                "sun_intensity", "sun_elevation", 
                                "sun_rotation", "altitude", "air_density", 
                                "dust_density", "ozone_density"],
        "ShaderNodeTexVoronoi" : ["voronoi_dimensions", "feature", "distance"],
        "ShaderNodeTexWave" : ["wave_type", "rings_direction", "wave_profile"],
        "ShaderNodeTexWhiteNoise" : ["noise_dimensions"],

        #color
        "ShaderNodeMix" : ["data_type", "clamp_factor", "factor_mode", "blend_type",
                            "clamp_result"],

        #vector
        "ShaderNodeBump" : ["invert"],
        "ShaderNodeDisplacement" : ["space"],
        "ShaderNodeMapping" : ["vector_type"],
        "ShaderNodeNormalMap" : ["space", "uv_map"],
        "ShaderNodeVectorDisplacement" : ["space"],
        "ShaderNodeVectorRotate" : ["rotation_type", "invert"],
        "ShaderNodeVectorTransform" : ["vector_type", "convert_from", "convert_to"],
        
        #converter
        "ShaderNodeClamp" : ["clamp_type"],
        "ShaderNodeCombineColor" : ["mode"],
        "ShaderNodeMapRange" : ["data_type", "interpolation_type", "clamp"],
        "ShaderNodeMath" : ["operation", "use_clamp"],
        "ShaderNodeSeparateColor" : ["mode"],
        "ShaderNodeVectorMath" : ["operation"]
        }

        curve_nodes = {'ShaderNodeFloatCurve', 
                    'ShaderNodeVectorCurve', 
                    'ShaderNodeRGBCurve'}

        image_nodes = {'ShaderNodeTexEnvironment',
                    'ShaderNodeTexImage'}

        #find node group to replicate
        nt = bpy.data.materials[self.material_name].node_tree
        if nt is None:
            self.report({'ERROR'},("ExportPreset: This doesn't seem to be a "
                                "valid material. Is Use Nodes selected?"))
            return {'CANCELLED'}

        #set up names to use in generated addon
        mat_var = clean_string(self.material_name)
        
        if self.mode == 'ADDON':
            dir = bpy.path.abspath(context.scene.preset_options.dir_path)
            if not dir or dir == "":
                self.report({'ERROR'},
                            ("ExportPreset: Save your blender file before using "
                            "ExportPreset!"))
                return {'CANCELLED'}

            zip_dir = os.path.join(dir, mat_var)
            addon_dir = os.path.join(zip_dir, mat_var)
            if not os.path.exists(addon_dir):
                os.makedirs(addon_dir)
            file = open(f"{addon_dir}/__init__.py", "w")

            create_header(file, self.material_name)
            class_name = clean_string(self.material_name, lower=False)
            init_operator(file, class_name, mat_var, self.material_name)

            file.write("\tdef execute(self, context):\n")
        else:
            file = StringIO("")

        #def create_material(indent: str):
        #    file.write((f"{indent}mat = bpy.data.materials.new("
        #                f"name = {str_to_py_str(self.material_name)})\n"))
        #    file.write(f"{indent}mat.use_nodes = True\n")
        #
        #if self.mode == 'ADDON':
        #    create_material("\t\t")
        #elif self.mode == 'SCRIPT':
        #    create_material("")

        #set to keep track of already created node trees
        node_trees = set()

        #dictionary to keep track of node->variable name pairs
        node_vars = {}

        #keeps track of all used variables
        used_vars = {}

        def is_outermost_node_group(level: int) -> bool:
            if self.mode == 'ADDON' and level == 2:
                return True
            elif self.mode == 'SCRIPT' and level == 0:
                return True
            return False

        def process_mat_node_group(node_tree, level, node_vars, used_vars):
            if is_outermost_node_group(level):
                nt_var = create_var(self.material_name, used_vars)
                nt_name = self.material_name
            else:
                nt_var = create_var(node_tree.name, used_vars)
                nt_name = node_tree.name

            outer, inner = make_indents(level)

            #initialize node group
            file.write(f"{outer}#initialize {nt_var} node group\n")
            file.write(f"{outer}def {nt_var}_node_group():\n")

            if is_outermost_node_group(level): #outermost node group
                file.write(f"{inner}{nt_var} = mat.node_tree\n")
                #file.write(f"{inner}#start with a clean node tree\n")
                #file.write(f"{inner}for node in {nt_var}.nodes:\n")
                #file.write(f"{inner}\t{nt_var}.nodes.remove(node)\n")
            else:
                file.write((f"{inner}if {str_to_py_str(nt_name)} in bpy.data.node_groups:\n"))
                file.write((f"{inner}    return\n"))
                file.write((f"{inner}else:\n"))
                file.write((f"{inner}    {nt_var}"
                        f"= bpy.data.node_groups.new("
                        f"type = \'ShaderNodeTree\', "
                        f"name = {str_to_py_str(nt_name)})\n"))
                file.write("\n")

            inputs_set = False
            outputs_set = False

            #initialize nodes
            file.write(f"{inner}#initialize {nt_var} nodes\n")

            #dictionary to keep track of node->variable name pairs
            node_vars = {}

            for node in node_tree.nodes:
                if node.bl_idname == 'ShaderNodeGroup':
                    node_nt = node.node_tree
                    if node_nt is not None and node_nt not in node_trees:
                        process_mat_node_group(node_nt, level + 1, node_vars, 
                                            used_vars)
                        node_trees.add(node_nt)
                
                node_var = create_node(node, file, inner, nt_var, node_vars, 
                                    used_vars)
                
                set_settings_defaults(node, node_settings, file, inner, node_var)
                hide_sockets(node, file, inner, node_var)

                if node.bl_idname == 'ShaderNodeGroup':
                    if node.node_tree is not None:
                        file.write((f"{inner}{node_var}.node_tree = "
                                    f"bpy.data.node_groups"
                                    f"[\"{node.node_tree.name}\"]\n"))
                elif node.bl_idname == 'NodeGroupInput' and not inputs_set:
                    group_io_settings(node, file, inner, "input", nt_var, node_tree)
                    inputs_set = True

                elif node.bl_idname == 'NodeGroupOutput' and not outputs_set:
                    group_io_settings(node, file, inner, "output", nt_var, node_tree)
                    outputs_set = True

                elif node.bl_idname in image_nodes and self.mode == 'ADDON':
                    img = node.image
                    if img is not None and img.source in {'FILE', 'GENERATED', 'TILED'}:
                        save_image(img, addon_dir)
                        load_image(img, file, inner, f"{node_var}.image")
                        image_user_settings(node, file, inner, node_var)

                elif node.bl_idname == 'ShaderNodeValToRGB':
                    color_ramp_settings(node, file, inner, node_var)

                elif node.bl_idname in curve_nodes:
                    curve_node_settings(node, file, inner, node_var)

                if self.mode == 'ADDON':
                    set_input_defaults(node, file, inner, node_var, addon_dir)
                else:
                    set_input_defaults(node, file, inner, node_var)
                set_output_defaults(node, file, inner, node_var)

            set_parents(node_tree, file, inner, node_vars)
            set_locations(node_tree, file, inner, node_vars)
            set_dimensions(node_tree, file, inner, node_vars)
            
            init_links(node_tree, file, inner, nt_var, node_vars)
            
            file.write(f"\n{outer}{nt_var}_node_group()\n\n")

        if self.mode == 'ADDON':
            level = 2
        else:
            level = 0        
        process_mat_node_group(nt, level, node_vars, used_vars)

        if self.mode == 'ADDON':
            file.write("\t\treturn {'FINISHED'}\n\n")

            create_menu_func(file, class_name)
            create_register_func(file, class_name)
            create_unregister_func(file, class_name)
            create_main_func(file)
        else:
            #context.window_manager.clipboard = file.getvalue()
            dir = bpy.path.abspath(context.scene.preset_options.dir_path)
            if not dir or dir == "":
                self.report({'ERROR'},
                            ("ExportPreset: Save your blender file before using "
                            "ExportPreset!"))
                return {'CANCELLED'}
            
            if not os.path.exists(dir):
                os.makedirs(dir)
            # 初始化檔案名稱和後綴
            filename = f"{self.material_name}.py"
            suffix = 0
            # 檢查檔案是否存在，如果存在，則增加後綴
            while os.path.isfile(f"{dir}/{filename}"):
                suffix += 1
                filename = f"{self.material_name}{suffix}.py"
            # 打開檔案並寫入內容
            myfile = open(f"{dir}/{filename}", "w", encoding="utf8")
            myfile.write(file.getvalue())
            myfile.close()
        
        file.close()
        
        if self.mode == 'ADDON':
            zip_addon(zip_dir)
        if self.mode == 'SCRIPT':
            #location = "clipboard"
            location = dir
        else:
            location = dir
        self.report({'INFO'}, f"ExportPreset: Saved material to {location}")
        return {'FINISHED'}

class SwitchPreset(Operator):
    bl_idname = 'mmd_tools.switch_material_preset'
    bl_label = 'Switch Preset'
    bl_description = 'Switch between multiple Material Outputs of selected objects.'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return next((x for x in context.selected_objects if x.type == 'MESH'), None)

    def execute(self, context):
        try:
            for obj in context.selected_objects:
                if obj.type != 'MESH':
                    continue
                for i in obj.material_slots:
                    if not i.material:
                        continue
                    
                    nodes = i.material.node_tree.nodes
                    matout_nodes = [node for node in nodes if node.type == 'OUTPUT_MATERIAL']
                    
                    for i in range(0, len(matout_nodes)):
                        if matout_nodes[i].is_active_output == True:
                            j = (i + 1) % len(matout_nodes)
                            matout_nodes[j].is_active_output = True
                            self.report({'INFO'}, f"switch to node : {matout_nodes[j].name}")
                            break
        except Exception as e:
            err_msg = traceback.format_exc()
            self.report({'ERROR'}, err_msg)
        return {'FINISHED'}

class ActivePreset(Operator):
    bl_idname = 'mmd_tools.active_material_preset'
    bl_label = 'Active Preset'
    bl_description = 'Active Material Outputs of selected objects with the same Material Output node name in Active object.'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return next((x for x in context.selected_objects if x.type == 'MESH'), None)

    def execute(self, context):
        nodes = context.active_object.active_material.node_tree.nodes
        name = ""
        matout_nodes = [node for node in nodes if node.type == 'OUTPUT_MATERIAL']
        for matout_node in matout_nodes:
            if matout_node.is_active_output == True:
                name = matout_node.name
                break
        if name == "":
            if matout_nodes != []:
                matout_nodes[0].is_active_output = True
                name = matout_nodes[0].name
            else:
                return {'CANCELLED'}
        self.report({'INFO'}, f"active node name : {name}")
        try:
            for obj in context.selected_objects:
                if obj.type != 'MESH':
                    continue
                for i in obj.material_slots:
                    if not i.material:
                        continue

                    nodes = i.material.node_tree.nodes

                    if name in nodes:
                        nodes[name].is_active_output = True
        except Exception as e:
            err_msg = traceback.format_exc()
            self.report({'ERROR'}, err_msg)
        return {'FINISHED'}

class _OpenTextureBase(object):
    """ Create a texture for mmd model material.
    """
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    filepath: StringProperty(
        name="File Path",
        description="Filepath used for importing the file",
        maxlen=1024,
        subtype='FILE_PATH',
        )

    use_filter_image: BoolProperty(
        default=True,
        options={'HIDDEN'},
        )

    def invoke(self, context, event):
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}

class OpenTexture(Operator, _OpenTextureBase):
    bl_idname = 'mmd_tools.material_open_texture'
    bl_label = 'Open Texture'
    bl_description = 'Create main texture of active material'

    def execute(self, context):
        mat = context.active_object.active_material
        fnMat = FnMaterial(mat)
        fnMat.create_texture(self.filepath)
        return {'FINISHED'}

class RemoveTexture(Operator):
    """ Create a texture for mmd model material.
    """
    bl_idname = 'mmd_tools.material_remove_texture'
    bl_label = 'Remove Texture'
    bl_description = 'Remove main texture of active material'
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        mat = context.active_object.active_material
        fnMat = FnMaterial(mat)
        fnMat.remove_texture()
        return {'FINISHED'}

class OpenSphereTextureSlot(Operator, _OpenTextureBase):
    """ Create a texture for mmd model material.
    """
    bl_idname = 'mmd_tools.material_open_sphere_texture'
    bl_label = 'Open Sphere Texture'
    bl_description = 'Create sphere texture of active material'

    def execute(self, context):
        mat = context.active_object.active_material
        fnMat = FnMaterial(mat)
        fnMat.create_sphere_texture(self.filepath, context.active_object)
        return {'FINISHED'}

class RemoveSphereTexture(Operator):
    """ Create a texture for mmd model material.
    """
    bl_idname = 'mmd_tools.material_remove_sphere_texture'
    bl_label = 'Remove Sphere Texture'
    bl_description = 'Remove sphere texture of active material'
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    def execute(self, context):
        mat = context.active_object.active_material
        fnMat = FnMaterial(mat)
        fnMat.remove_sphere_texture()
        return {'FINISHED'}

class MoveMaterialUp(Operator):
    bl_idname = 'mmd_tools.move_material_up'
    bl_label = 'Move Material Up'
    bl_description = 'Moves selected material one slot up'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        valid_mesh = obj and obj.type == 'MESH' and obj.mmd_type == 'NONE'
        return valid_mesh and obj.active_material_index > 0

    def execute(self, context):
        obj = context.active_object
        current_idx = obj.active_material_index
        prev_index = current_idx - 1
        try:
            FnMaterial.swap_materials(obj, current_idx, prev_index,
                                      reverse=True, swap_slots=True)
        except MaterialNotFoundError:
            self.report({'ERROR'}, 'Materials not found')
            return { 'CANCELLED' }
        obj.active_material_index = prev_index

        return { 'FINISHED' }

class MoveMaterialDown(Operator):
    bl_idname = 'mmd_tools.move_material_down'
    bl_label = 'Move Material Down'
    bl_description = 'Moves the selected material one slot down'
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        valid_mesh = obj and obj.type == 'MESH' and obj.mmd_type == 'NONE'
        return valid_mesh and obj.active_material_index < len(obj.material_slots) - 1

    def execute(self, context):
        obj = context.active_object
        current_idx = obj.active_material_index
        next_index = current_idx + 1
        try:
            FnMaterial.swap_materials(obj, current_idx, next_index,
                                      reverse=True, swap_slots=True)
        except MaterialNotFoundError:
            self.report({'ERROR'}, 'Materials not found')
            return { 'CANCELLED' }
        obj.active_material_index = next_index
        return { 'FINISHED' }

class EdgePreviewSetup(Operator):
    bl_idname = 'mmd_tools.edge_preview_setup'
    bl_label = 'Edge Preview Setup'
    bl_description = 'Preview toon edge settings of active model using "Solidify" modifier'
    bl_options = {'REGISTER', 'UNDO', 'INTERNAL'}

    action: bpy.props.EnumProperty(
        name='Action',
        description='Select action',
        items=[
            ('CREATE', 'Create', 'Create toon edge', 0),
            ('CLEAN', 'Clean', 'Clear toon edge', 1),
            ],
        default='CREATE',
        )

    def execute(self, context):
        from mmd_tools.core.model import Model
        root = Model.findRoot(context.active_object)
        if root is None:
            self.report({'ERROR'}, 'Select a MMD model')
            return {'CANCELLED'}

        rig = Model(root)
        if self.action == 'CLEAN':
            for obj in rig.meshes():
                self.__clean_toon_edge(obj)
        else:
            from mmd_tools.bpyutils import Props
            scale = 0.2*getattr(rig.rootObject(), Props.empty_display_size)
            counts = sum(self.__create_toon_edge(obj, scale) for obj in rig.meshes())
            self.report({'INFO'}, 'Created %d toon edge(s)'%counts)
        return {'FINISHED'}

    def __clean_toon_edge(self, obj):
        if 'mmd_edge_preview' in obj.modifiers:
            obj.modifiers.remove(obj.modifiers['mmd_edge_preview'])

        if 'mmd_edge_preview' in obj.vertex_groups:
            obj.vertex_groups.remove(obj.vertex_groups['mmd_edge_preview'])

        FnMaterial.clean_materials(obj, can_remove=lambda m: m and m.name.startswith('mmd_edge.'))

    def __create_toon_edge(self, obj, scale=1.0):
        self.__clean_toon_edge(obj)
        materials = obj.data.materials
        material_offset = len(materials)
        for m in tuple(materials):
            if m and m.mmd_material.enabled_toon_edge:
                mat_edge = self.__get_edge_material('mmd_edge.'+m.name, m.mmd_material.edge_color, materials)
                materials.append(mat_edge)
            elif material_offset > 1:
                mat_edge = self.__get_edge_material('mmd_edge.disabled', (0, 0, 0, 0), materials)
                materials.append(mat_edge)
        if len(materials) > material_offset:
            mod = obj.modifiers.get('mmd_edge_preview', None)
            if mod is None:
                mod = obj.modifiers.new('mmd_edge_preview', 'SOLIDIFY')
            mod.material_offset = material_offset
            mod.thickness_vertex_group = 1e-3 # avoid overlapped faces
            mod.use_flip_normals = True
            mod.use_rim = False
            mod.offset = 1
            self.__create_edge_preview_group(obj)
            mod.thickness = scale
            mod.vertex_group = 'mmd_edge_preview'
        return len(materials) - material_offset

    def __create_edge_preview_group(self, obj):
        vertices, materials = obj.data.vertices, obj.data.materials
        weight_map = {i:m.mmd_material.edge_weight for i, m in enumerate(materials) if m}
        scale_map = {}
        vg_scale_index = obj.vertex_groups.find('mmd_edge_scale')
        if vg_scale_index >= 0:
            scale_map = {v.index:g.weight for v in vertices for g in v.groups if g.group == vg_scale_index}
        vg_edge_preview = obj.vertex_groups.new(name='mmd_edge_preview')
        for i, mi in {v:f.material_index for f in reversed(obj.data.polygons) for v in f.vertices}.items():
            weight = scale_map.get(i, 1.0) * weight_map.get(mi, 1.0) * 0.02
            vg_edge_preview.add(index=[i], weight=weight, type='REPLACE')

    def __get_edge_material(self, mat_name, edge_color, materials):
        if mat_name in materials:
            return materials[mat_name]
        mat = bpy.data.materials.get(mat_name, None)
        if mat is None:
            mat = bpy.data.materials.new(mat_name)
        mmd_mat = mat.mmd_material
        # note: edge affects ground shadow
        mmd_mat.is_double_sided = mmd_mat.enabled_drop_shadow = False
        mmd_mat.enabled_self_shadow_map = mmd_mat.enabled_self_shadow = False
        #mmd_mat.enabled_self_shadow_map = True # for blender 2.78+ BI viewport only
        mmd_mat.diffuse_color = mmd_mat.specular_color = (0, 0, 0)
        mmd_mat.ambient_color = edge_color[:3]
        mmd_mat.alpha = edge_color[3]
        mmd_mat.edge_color = edge_color
        self.__make_shader(mat)
        return mat

    def __make_shader(self, m):
        m.use_nodes = True
        nodes, links = m.node_tree.nodes, m.node_tree.links

        node_shader = nodes.get('mmd_edge_preview', None)
        if node_shader is None or not any(s.is_linked for s in node_shader.outputs):
            XPOS, YPOS = 210, 110
            nodes.clear()
            node_shader = nodes.new('ShaderNodeGroup')
            node_shader.name = 'mmd_edge_preview'
            node_shader.location = (0, 0)
            node_shader.width = 200
            node_shader.node_tree = self.__get_edge_preview_shader()

            if bpy.app.version < (2, 80, 0):
                node_out = nodes.new('ShaderNodeOutput')
                node_out.location = (XPOS*2, YPOS*2)
                links.new(node_shader.outputs['Color'], node_out.inputs['Color'])
                links.new(node_shader.outputs['Alpha'], node_out.inputs['Alpha'])

            node_out = nodes.new('ShaderNodeOutputMaterial')
            node_out.location = (XPOS*2, YPOS*0)
            links.new(node_shader.outputs['Shader'], node_out.inputs['Surface'])

        node_shader.inputs['Color'].default_value = m.mmd_material.edge_color
        node_shader.inputs['Alpha'].default_value = m.mmd_material.edge_color[3]

    def __get_edge_preview_shader(self):
        group_name = 'MMDEdgePreview'
        shader = bpy.data.node_groups.get(group_name, None) or bpy.data.node_groups.new(name=group_name, type='ShaderNodeTree')
        if len(shader.nodes):
            return shader

        ng = _NodeGroupUtils(shader)

        node_input = ng.new_node('NodeGroupInput', (-5, 0))
        node_output = ng.new_node('NodeGroupOutput', (3, 0))

        ############################################################################
        node_color = ng.new_node('ShaderNodeMixRGB', (-1, -1.5))
        node_color.mute = True

        ng.new_input_socket('Color', node_color.inputs['Color1'])

        if bpy.app.version < (2, 80, 0):
            node_geo = ng.new_node('ShaderNodeGeometry', (-2, -2.5))
            node_cull = ng.new_math_node('MULTIPLY', (-1, -2.5))

            ng.links.new(node_geo.outputs['Front/Back'], node_cull.inputs[1])

            ng.new_input_socket('Alpha', node_cull.inputs[0])
            ng.new_output_socket('Color', node_color.outputs['Color'])
            ng.new_output_socket('Alpha', node_cull.outputs['Value'])

        ############################################################################
        node_ray = ng.new_node('ShaderNodeLightPath', (-3, 1.5))
        node_geo = ng.new_node('ShaderNodeNewGeometry', (-3, 0))
        node_max = ng.new_math_node('MAXIMUM', (-2, 1.5))
        node_max.mute = True
        node_gt = ng.new_math_node('GREATER_THAN', (-1, 1))
        node_alpha = ng.new_math_node('MULTIPLY', (0, 1))
        node_trans = ng.new_node('ShaderNodeBsdfTransparent', (0, 0))
        EDGE_NODE_NAME = 'ShaderNodeEmission' if bpy.app.version < (2, 80, 0) else 'ShaderNodeBackground'
        node_rgb = ng.new_node(EDGE_NODE_NAME, (0, -0.5)) # BsdfDiffuse/Background/Emission
        node_mix = ng.new_node('ShaderNodeMixShader', (1, 0.5))

        links = ng.links
        links.new(node_ray.outputs['Is Camera Ray'], node_max.inputs[0])
        links.new(node_ray.outputs['Is Glossy Ray'], node_max.inputs[1])
        links.new(node_max.outputs['Value'], node_gt.inputs[0])
        links.new(node_geo.outputs['Backfacing'], node_gt.inputs[1])
        links.new(node_gt.outputs['Value'], node_alpha.inputs[0])
        links.new(node_alpha.outputs['Value'], node_mix.inputs['Fac'])
        links.new(node_trans.outputs['BSDF'], node_mix.inputs[1])
        links.new(node_rgb.outputs[0], node_mix.inputs[2])
        links.new(node_color.outputs['Color'], node_rgb.inputs['Color'])

        ng.new_input_socket('Alpha', node_alpha.inputs[1])
        ng.new_output_socket('Shader', node_mix.outputs['Shader'])

        return shader
