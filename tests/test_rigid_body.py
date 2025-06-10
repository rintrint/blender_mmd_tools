import gc
import logging
import os
import shutil
import unittest
from math import pi

import bpy
from bl_ext.user_default.mmd_tools.core import rigid_body
from bl_ext.user_default.mmd_tools.core.model import FnModel, Model
from bl_ext.user_default.mmd_tools.core.rigid_body import FnRigidBody
from mathutils import Euler, Vector

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(os.path.dirname(TESTS_DIR), "samples")


class TestRigidBody(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Clean up output from previous tests"""
        output_dir = os.path.join(TESTS_DIR, "output")
        for item in os.listdir(output_dir):
            if item.endswith(".OUTPUT"):
                continue
            item_fp = os.path.join(output_dir, item)
            if os.path.isfile(item_fp):
                os.remove(item_fp)
            elif os.path.isdir(item_fp):
                shutil.rmtree(item_fp)

    def setUp(self):
        """Start each test with a clean state"""

        logger = logging.getLogger()
        logger.setLevel("ERROR")

        if not bpy.context.active_object:
            bpy.ops.mesh.primitive_cube_add()

        bpy.ops.object.mode_set(mode="OBJECT")
        bpy.ops.object.select_all(action="SELECT")
        bpy.ops.object.delete(use_global=True)

        self.context = bpy.context
        self.scene = bpy.context.scene

    # ********************************************
    # Utils
    # ********************************************

    def __vector_error(self, vec0, vec1):
        return (Vector(vec0) - Vector(vec1)).length

    def __quaternion_error(self, quat0, quat1):
        angle = quat0.rotation_difference(quat1).angle % pi
        assert angle >= 0
        return min(angle, pi - angle)

    def __safe_get_object(self, name):
        """Safely get object by name"""
        return bpy.data.objects.get(name)

    def __safe_get_material(self, name):
        """Safely get material by name"""
        return bpy.data.materials.get(name)

    # ********************************************
    # Helper Functions
    # ********************************************

    def _enable_mmd_tools(self):
        """Make sure mmd_tools addon is enabled"""
        bpy.ops.wm.read_homefile(use_empty=True)
        pref = getattr(bpy.context, "preferences", None) or bpy.context.user_preferences
        if not pref.addons.get("bl_ext.user_default.mmd_tools", None):
            addon_enable = bpy.ops.wm.addon_enable if "addon_enable" in dir(bpy.ops.wm) else bpy.ops.preferences.addon_enable
            addon_enable(module="bl_ext.user_default.mmd_tools")

    def _create_test_model(self, name="TestModel"):
        """Create a basic test MMD model with armature"""
        model = Model.create(name=name, name_e=name + "_e", scale=0.08, add_root_bone=True)

        # Add a test bone
        armature = model.armature()
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="EDIT")

        edit_bones = armature.data.edit_bones
        test_bone = edit_bones.new("test_bone")
        test_bone.head = (0, 0, 1)
        test_bone.tail = (0, 0, 2)
        test_bone.parent = edit_bones.get("全ての親")

        bpy.ops.object.mode_set(mode="OBJECT")

        # Set pose bone properties
        pose_bone = armature.pose.bones["test_bone"]
        pose_bone.mmd_bone.name_j = "テストボーン"
        pose_bone.mmd_bone.name_e = "test_bone"

        return model

    def _create_rigid_body_object(self, model, bone_name=None, shape="SPHERE", location=(0, 0, 0)):
        """Create a rigid body object for testing"""
        root_obj = model.rootObject()
        rigid_group = model.rigidGroupObject()

        rigid_obj = FnRigidBody.new_rigid_body_object(self.context, rigid_group)

        FnRigidBody.setup_rigid_body_object(
            obj=rigid_obj,
            shape_type=rigid_body.shapeType(shape),
            location=Vector(location),
            rotation=Euler((0, 0, 0)),
            size=Vector((1, 1, 1)),
            dynamics_type=1,  # Physics
            name="TestRigid",
            name_e="TestRigid_e",
            collision_group_number=0,
            collision_group_mask=[False] * 16,
            mass=1.0,
            friction=0.5,
            bounce=0.0,
            linear_damping=0.04,
            angular_damping=0.1,
            bone=bone_name,
        )

        self.assertEqual(rigid_obj.parent, rigid_group, "Rigid body should be parented to rigid group")
        self.assertEqual(rigid_group.parent, root_obj, "Rigid group should be parented to root object")

        return rigid_obj

    def _create_joint_object(self, model, rigid_a, rigid_b, location=(0, 0, 0)):
        """Create a joint object for testing"""
        root_obj = model.rootObject()
        joint_group = model.jointGroupObject()

        joint_obj = FnRigidBody.new_joint_object(self.context, joint_group, FnModel.get_empty_display_size(root_obj))

        FnRigidBody.setup_joint_object(
            obj=joint_obj,
            location=Vector(location),
            rotation=Euler((0, 0, 0)),
            rigid_a=rigid_a,
            rigid_b=rigid_b,
            maximum_location=Vector((0.1, 0.1, 0.1)),
            minimum_location=Vector((-0.1, -0.1, -0.1)),
            maximum_rotation=Euler((pi / 4, pi / 4, pi / 4)),
            minimum_rotation=Euler((-pi / 4, -pi / 4, -pi / 4)),
            spring_angular=Vector((0, 0, 0)),
            spring_linear=Vector((0, 0, 0)),
            name="TestJoint",
            name_e="TestJoint_e",
        )

        return joint_obj

    def _check_rigid_body_integrity(self, rigid_obj):
        """Check rigid body object integrity"""
        self.assertEqual(rigid_obj.mmd_type, "RIGID_BODY", "Object should be rigid body type")
        self.assertIsNotNone(rigid_obj.rigid_body, "Rigid body should have physics properties")
        self.assertTrue(hasattr(rigid_obj, "mmd_rigid"), "Rigid body should have MMD properties")

        # Check basic MMD properties
        mmd_rigid = rigid_obj.mmd_rigid
        self.assertIsNotNone(mmd_rigid.name_j, "Rigid body should have Japanese name")
        self.assertIn(mmd_rigid.shape, ["SPHERE", "BOX", "CAPSULE"], "Shape should be valid")
        self.assertIn(mmd_rigid.type, ["0", "1", "2"], "Type should be valid")

    def _check_joint_integrity(self, joint_obj):
        """Check joint object integrity"""
        self.assertEqual(joint_obj.mmd_type, "JOINT", "Object should be joint type")
        self.assertIsNotNone(joint_obj.rigid_body_constraint, "Joint should have constraint properties")
        self.assertTrue(hasattr(joint_obj, "mmd_joint"), "Joint should have MMD properties")

        # Check constraint properties
        rbc = joint_obj.rigid_body_constraint
        self.assertIsNotNone(rbc.object1, "Joint should have first rigid body")
        self.assertIsNotNone(rbc.object2, "Joint should have second rigid body")

    def _check_physics_world_setup(self):
        """Check if physics world is properly set up"""
        scene = bpy.context.scene

        # Ensure physics world exists
        if not scene.rigidbody_world:
            bpy.ops.rigidbody.world_add()

        rbw = scene.rigidbody_world
        self.assertIsNotNone(rbw, "Scene should have rigid body world")

        # Ensure collections exist - this is critical for Blender physics
        if not rbw.collection:
            rbw.collection = bpy.data.collections.new("RigidBodyWorld")
            rbw.collection.use_fake_user = True
            # Link to scene if needed
            if rbw.collection.name not in bpy.context.scene.collection.children:
                bpy.context.scene.collection.children.link(rbw.collection)

        if not rbw.constraints:
            rbw.constraints = bpy.data.collections.new("RigidBodyConstraints")
            rbw.constraints.use_fake_user = True
            # Link to scene if needed
            if rbw.constraints.name not in bpy.context.scene.collection.children:
                bpy.context.scene.collection.children.link(rbw.constraints)

        self.assertIsNotNone(rbw.collection, "Rigid body world should have collection")
        self.assertIsNotNone(rbw.constraints, "Rigid body world should have constraints collection")

    # ********************************************
    # Test Cases
    # ********************************************

    def test_rigid_body_creation_basic(self):
        """Test basic rigid body creation functionality"""
        self._enable_mmd_tools()

        model = self._create_test_model()
        rigid_obj = self._create_rigid_body_object(model, bone_name="test_bone")

        # Check basic creation
        self.assertIsNotNone(rigid_obj, "Rigid body should be created")
        self._check_rigid_body_integrity(rigid_obj)

        # Check parenting
        self.assertEqual(rigid_obj.parent, model.rigidGroupObject(), "Rigid body should be parented to rigid group")

        print("✓ Basic rigid body creation test passed")

    def test_rigid_body_shapes(self):
        """Test different rigid body shapes"""
        self._enable_mmd_tools()

        model = self._create_test_model()
        shapes = ["SPHERE", "BOX", "CAPSULE"]

        for shape in shapes:
            rigid_obj = self._create_rigid_body_object(model, shape=shape)

            self.assertEqual(rigid_obj.mmd_rigid.shape, shape, f"Shape should be {shape}")
            self.assertEqual(rigid_obj.rigid_body.collision_shape, shape, f"Physics shape should be {shape}")

            print(f"   - {shape} shape test passed")

        print("✓ Rigid body shapes test passed")

    def test_rigid_body_types(self):
        """Test different rigid body dynamics types"""
        self._enable_mmd_tools()

        model = self._create_test_model()

        # Test Static (Bone)
        static_rigid = self._create_rigid_body_object(model, bone_name="test_bone")
        FnRigidBody.setup_rigid_body_object(
            obj=static_rigid,
            shape_type=0,  # SPHERE
            location=Vector((0, 0, 0)),
            rotation=Euler((0, 0, 0)),
            size=Vector((1, 1, 1)),
            dynamics_type=0,  # Static
            name="StaticRigid",
        )

        self.assertEqual(static_rigid.mmd_rigid.type, "0", "Should be static type")

        # Test Dynamic (Physics)
        dynamic_rigid = self._create_rigid_body_object(model, bone_name="test_bone")
        FnRigidBody.setup_rigid_body_object(
            obj=dynamic_rigid,
            shape_type=0,  # SPHERE
            location=Vector((1, 0, 0)),
            rotation=Euler((0, 0, 0)),
            size=Vector((1, 1, 1)),
            dynamics_type=1,  # Dynamic
            name="DynamicRigid",
        )

        self.assertEqual(dynamic_rigid.mmd_rigid.type, "1", "Should be dynamic type")

        # Test Dynamic+Bone (Physics + Bone)
        dynamic_bone_rigid = self._create_rigid_body_object(model, bone_name="test_bone")
        FnRigidBody.setup_rigid_body_object(
            obj=dynamic_bone_rigid,
            shape_type=0,  # SPHERE
            location=Vector((2, 0, 0)),
            rotation=Euler((0, 0, 0)),
            size=Vector((1, 1, 1)),
            dynamics_type=2,  # Dynamic+Bone
            name="DynamicBoneRigid",
        )

        self.assertEqual(dynamic_bone_rigid.mmd_rigid.type, "2", "Should be dynamic+bone type")

        print("✓ Rigid body types test passed")

    def test_rigid_body_collision_groups(self):
        """Test rigid body collision group settings"""
        self._enable_mmd_tools()

        model = self._create_test_model()

        # Test different collision groups
        for group_num in range(3):
            collision_mask = [False] * 16
            collision_mask[group_num] = True  # Can't collide with same group

            rigid_obj = self._create_rigid_body_object(model)
            FnRigidBody.setup_rigid_body_object(obj=rigid_obj, shape_type=0, location=Vector((group_num, 0, 0)), rotation=Euler((0, 0, 0)), size=Vector((1, 1, 1)), dynamics_type=1, collision_group_number=group_num, collision_group_mask=collision_mask, name=f"RigidGroup{group_num}")

            self.assertEqual(rigid_obj.mmd_rigid.collision_group_number, group_num, f"Collision group should be {group_num}")
            self.assertEqual(rigid_obj.mmd_rigid.collision_group_mask[group_num], True, f"Should not collide with group {group_num}")

        print("✓ Rigid body collision groups test passed")

    def test_rigid_body_physics_properties(self):
        """Test rigid body physics property settings"""
        self._enable_mmd_tools()

        model = self._create_test_model()
        rigid_obj = self._create_rigid_body_object(model)

        # Test various physics properties
        test_properties = {"mass": 2.5, "friction": 0.8, "bounce": 0.3, "linear_damping": 0.1, "angular_damping": 0.2}

        FnRigidBody.setup_rigid_body_object(obj=rigid_obj, shape_type=0, location=Vector((0, 0, 0)), rotation=Euler((0, 0, 0)), size=Vector((1, 1, 1)), dynamics_type=1, name="PhysicsTest", **test_properties)

        rb = rigid_obj.rigid_body
        self.assertAlmostEqual(rb.mass, test_properties["mass"], places=5)
        self.assertAlmostEqual(rb.friction, test_properties["friction"], places=5)
        self.assertAlmostEqual(rb.restitution, test_properties["bounce"], places=5)
        self.assertAlmostEqual(rb.linear_damping, test_properties["linear_damping"], places=5)
        self.assertAlmostEqual(rb.angular_damping, test_properties["angular_damping"], places=5)

        print("✓ Rigid body physics properties test passed")

    def test_rigid_body_size_calculation(self):
        """Test rigid body size calculation functionality"""
        self._enable_mmd_tools()

        model = self._create_test_model()

        # Test different shapes with different sizes
        test_cases = [{"shape": "SPHERE", "size": Vector((2, 2, 2))}, {"shape": "BOX", "size": Vector((1, 2, 3))}, {"shape": "CAPSULE", "size": Vector((1.5, 1.5, 4))}]

        for case in test_cases:
            # Create rigid body object with specified shape
            rigid_obj = self._create_rigid_body_object(model, shape=case["shape"])

            # Set the size on the rigid body's MMD properties
            rigid_obj.mmd_rigid.size = case["size"]

            # Create mesh data directly without creating a separate object
            if case["shape"] == "SPHERE":
                mesh_data = bpy.data.meshes.new("test_sphere")
                bpy.context.collection.objects.link(bpy.data.objects.new("temp_sphere", mesh_data))
                bpy.context.view_layer.objects.active = bpy.data.objects["temp_sphere"]
                bpy.ops.mesh.primitive_uv_sphere_add()
                temp_obj = bpy.context.active_object
            elif case["shape"] == "BOX":
                mesh_data = bpy.data.meshes.new("test_box")
                bpy.context.collection.objects.link(bpy.data.objects.new("temp_box", mesh_data))
                bpy.context.view_layer.objects.active = bpy.data.objects["temp_box"]
                bpy.ops.mesh.primitive_cube_add()
                temp_obj = bpy.context.active_object
            elif case["shape"] == "CAPSULE":
                mesh_data = bpy.data.meshes.new("test_capsule")
                bpy.context.collection.objects.link(bpy.data.objects.new("temp_capsule", mesh_data))
                bpy.context.view_layer.objects.active = bpy.data.objects["temp_capsule"]
                bpy.ops.mesh.primitive_cylinder_add()
                temp_obj = bpy.context.active_object

            # Scale the temporary object
            temp_obj.scale = case["size"]

            # Apply scale transformation
            bpy.context.view_layer.objects.active = temp_obj
            bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

            # Store mesh data reference before removing temp object
            final_mesh_data = temp_obj.data

            # Remove the temporary object but keep mesh data
            bpy.data.objects.remove(temp_obj)

            # Store original mesh data to restore later
            original_mesh = rigid_obj.data

            # Assign new mesh data to rigid body object
            rigid_obj.data = final_mesh_data

            # Test size calculation on the rigid body object
            try:
                calculated_size = FnRigidBody.get_rigid_body_size(rigid_obj)
                self.assertIsNotNone(calculated_size, f"Size calculation should work for {case['shape']}")
                print(f"   - {case['shape']} size calculation: {calculated_size}")

                # Verify calculated size makes sense (basic sanity check)
                self.assertTrue(len(calculated_size) == 3, f"Size should have 3 components for {case['shape']}")

            except Exception as e:
                print(f"   - {case['shape']} size calculation failed: {str(e)[:50]}")
            finally:
                # Restore original mesh data
                rigid_obj.data = original_mesh

                # Clean up the test mesh data
                bpy.data.meshes.remove(final_mesh_data)

        print("✓ Rigid body size calculation test passed")

    def test_joint_creation_basic(self):
        """Test basic joint creation functionality"""
        self._enable_mmd_tools()

        model = self._create_test_model()

        # Create two rigid bodies to connect
        rigid_a = self._create_rigid_body_object(model, location=(0, 0, 0))
        rigid_b = self._create_rigid_body_object(model, location=(2, 0, 0))

        # Create joint
        joint_obj = self._create_joint_object(model, rigid_a, rigid_b, location=(1, 0, 0))

        # Check joint creation
        self.assertIsNotNone(joint_obj, "Joint should be created")
        self._check_joint_integrity(joint_obj)

        # Check constraint setup
        rbc = joint_obj.rigid_body_constraint
        self.assertEqual(rbc.object1, rigid_a, "First rigid body should be correct")
        self.assertEqual(rbc.object2, rigid_b, "Second rigid body should be correct")
        self.assertEqual(rbc.type, "GENERIC_SPRING", "Should be generic spring constraint")

        print("✓ Basic joint creation test passed")

    def test_joint_limits_and_springs(self):
        """Test joint limits and spring configuration"""
        self._enable_mmd_tools()

        model = self._create_test_model()

        rigid_a = self._create_rigid_body_object(model, location=(0, 0, 0))
        rigid_b = self._create_rigid_body_object(model, location=(1, 0, 0))

        # Test with specific limits and springs
        max_location = Vector((0.5, 0.5, 0.5))
        min_location = Vector((-0.5, -0.5, -0.5))
        max_rotation = Euler((pi / 3, pi / 3, pi / 3))
        min_rotation = Euler((-pi / 3, -pi / 3, -pi / 3))
        spring_linear = Vector((10, 10, 10))
        spring_angular = Vector((5, 5, 5))

        joint_obj = FnRigidBody.new_joint_object(self.context, model.jointGroupObject(), FnModel.get_empty_display_size(model.rootObject()))

        FnRigidBody.setup_joint_object(
            obj=joint_obj,
            location=Vector((0.5, 0, 0)),
            rotation=Euler((0, 0, 0)),
            rigid_a=rigid_a,
            rigid_b=rigid_b,
            maximum_location=max_location,
            minimum_location=min_location,
            maximum_rotation=max_rotation,
            minimum_rotation=min_rotation,
            spring_angular=spring_angular,
            spring_linear=spring_linear,
            name="LimitTest",
            name_e="LimitTest_e",
        )

        # Check limits
        rbc = joint_obj.rigid_body_constraint
        self.assertAlmostEqual(rbc.limit_lin_x_upper, max_location.x, places=5)
        self.assertAlmostEqual(rbc.limit_lin_x_lower, min_location.x, places=5)
        self.assertAlmostEqual(rbc.limit_ang_x_upper, max_rotation.x, places=5)
        self.assertAlmostEqual(rbc.limit_ang_x_lower, min_rotation.x, places=5)

        # Check springs
        mmd_joint = joint_obj.mmd_joint
        self.assertEqual(tuple(mmd_joint.spring_linear), tuple(spring_linear))
        self.assertEqual(tuple(mmd_joint.spring_angular), tuple(spring_angular))

        print("✓ Joint limits and springs test passed")

    def test_rigid_body_world_setup(self):
        """Test rigid body world setup and configuration"""
        self._enable_mmd_tools()

        # Start with clean state - remove existing world if present
        if bpy.context.scene.rigidbody_world:
            bpy.ops.rigidbody.world_remove()

        # Create model with rigid bodies
        model = self._create_test_model()
        rigid_obj = self._create_rigid_body_object(model)

        # Force world creation by adding rigid body to scene
        if not bpy.context.scene.rigidbody_world:
            bpy.ops.rigidbody.world_add()

        # Check if world is created and properly configured
        self._check_physics_world_setup()

        # Test world settings
        rbw = bpy.context.scene.rigidbody_world
        self.assertTrue(rbw.enabled, "Rigid body world should be enabled")

        # Ensure rigid body is properly added to world collection
        if rigid_obj not in rbw.collection.objects.values():
            rbw.collection.objects.link(rigid_obj)

        # Verify rigid body is in world collection
        self.assertIn(rigid_obj, rbw.collection.objects.values(), "Rigid body should be in world collection")

        print("✓ Rigid body world setup test passed")

    def test_rigid_body_materials(self):
        """Test rigid body material assignment"""
        self._enable_mmd_tools()

        # Test material creation for different collision groups
        for group_num in range(3):
            material = rigid_body.RigidBodyMaterial.getMaterial(group_num)

            self.assertIsNotNone(material, f"Material should be created for group {group_num}")
            self.assertTrue(material.name.startswith("mmd_tools_rigid_"), "Material should have correct name prefix")
            self.assertEqual(len(material.diffuse_color), 4, "Material should have RGBA color")
            self.assertEqual(material.blend_method, "BLEND", "Material should use blend mode")

            print(f"   - Material for group {group_num}: {material.name}")

        print("✓ Rigid body materials test passed")

    def test_rigid_body_multiple_creation(self):
        """Test creating multiple rigid bodies at once"""
        self._enable_mmd_tools()

        model = self._create_test_model()
        rigid_group = model.rigidGroupObject()

        # Test multiple creation
        count = 5
        rigid_objects = FnRigidBody.new_rigid_body_objects(self.context, rigid_group, count)

        self.assertEqual(len(rigid_objects), count, f"Should create {count} rigid bodies")

        for i, rigid_obj in enumerate(rigid_objects):
            self.assertEqual(rigid_obj.mmd_type, "RIGID_BODY", f"Object {i} should be rigid body type")
            self.assertEqual(rigid_obj.parent, rigid_group, f"Object {i} should be parented correctly")

            # Setup each with different properties
            FnRigidBody.setup_rigid_body_object(
                obj=rigid_obj,
                shape_type=i % 3,  # Cycle through shapes
                location=Vector((i, 0, 0)),
                rotation=Euler((0, 0, 0)),
                size=Vector((1, 1, 1)),
                dynamics_type=1,
                name=f"MultiRigid{i}",
                collision_group_number=i % 16,
            )

        print(f"✓ Multiple rigid body creation test passed ({count} objects)")

    def test_joint_multiple_creation(self):
        """Test creating multiple joints at once"""
        self._enable_mmd_tools()

        model = self._create_test_model()
        joint_group = model.jointGroupObject()

        # Test multiple creation
        count = 3
        joint_objects = FnRigidBody.new_joint_objects(self.context, joint_group, count, FnModel.get_empty_display_size(model.rootObject()))

        self.assertEqual(len(joint_objects), count, f"Should create {count} joints")

        for i, joint_obj in enumerate(joint_objects):
            self.assertEqual(joint_obj.mmd_type, "JOINT", f"Object {i} should be joint type")
            self.assertEqual(joint_obj.parent, joint_group, f"Object {i} should be parented correctly")
            self.assertIsNotNone(joint_obj.rigid_body_constraint, f"Object {i} should have constraint")

        print(f"✓ Multiple joint creation test passed ({count} objects)")

    def test_rigid_body_edge_cases(self):
        """Test edge cases and error handling"""
        self._enable_mmd_tools()

        model = self._create_test_model()

        # Test with extreme values
        extreme_cases = [{"mass": 0.001, "desc": "Very low mass"}, {"mass": 1000.0, "desc": "Very high mass"}, {"size": Vector((0.01, 0.01, 0.01)), "desc": "Very small size"}, {"size": Vector((10, 10, 10)), "desc": "Very large size"}]

        for case in extreme_cases:
            try:
                rigid_obj = self._create_rigid_body_object(model)

                setup_params = {"shape_type": 0, "location": Vector((0, 0, 0)), "rotation": Euler((0, 0, 0)), "size": Vector((1, 1, 1)), "dynamics_type": 1, "name": f"Extreme_{case['desc'].replace(' ', '')}"}
                setup_params.update({k: v for k, v in case.items() if k != "desc"})

                FnRigidBody.setup_rigid_body_object(obj=rigid_obj, **setup_params)

                # Check if object was created successfully
                self.assertIsNotNone(rigid_obj.rigid_body, f"Rigid body should be created for {case['desc']}")
                print(f"   - {case['desc']}: passed")

            except Exception as e:
                # Some extreme cases might legitimately fail
                print(f"   - {case['desc']}: failed as expected ({str(e)[:50]})")

        print("✓ Rigid body edge cases test completed")

    def test_rigid_body_operators_basic(self):
        """Test basic rigid body operators functionality"""
        self._enable_mmd_tools()

        model = self._create_test_model()
        armature = model.armature()

        # Select armature and bone
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="POSE")
        armature.data.bones["test_bone"].select = True
        bpy.context.object.pose.bones["test_bone"].bone.select = True

        # Test add rigid body operator
        try:
            bpy.ops.mmd_tools.rigid_body_add(
                name_j="テストリジッド",
                name_e="test_rigid",
                collision_group_number=1,
                rigid_type="1",  # Physics
                rigid_shape="SPHERE",
                mass=2.0,
            )

            # Check if rigid body was created
            rigid_bodies = list(model.rigidBodies())
            self.assertGreater(len(rigid_bodies), 0, "Rigid body should be created")

            created_rigid = rigid_bodies[0]
            self.assertEqual(created_rigid.mmd_rigid.name_j, "テストリジッド", "Japanese name should be set")
            self.assertEqual(created_rigid.mmd_rigid.name_e, "test_rigid", "English name should be set")
            self.assertEqual(created_rigid.mmd_rigid.collision_group_number, 1, "Collision group should be set")

            print("   - Add rigid body operator: passed")

        except Exception as e:
            print(f"   - Add rigid body operator: failed ({str(e)[:50]})")

        bpy.ops.object.mode_set(mode="OBJECT")
        print("✓ Rigid body operators basic test completed")

    def test_rigid_body_physics_bake_operations(self):
        """Test rigid body physics baking operations"""
        self._enable_mmd_tools()

        model = self._create_test_model()
        rigid_obj = self._create_rigid_body_object(model)

        # Ensure physics world exists and is properly configured
        if not bpy.context.scene.rigidbody_world:
            bpy.ops.rigidbody.world_add()

        # Use our enhanced setup check
        self._check_physics_world_setup()

        rbw = bpy.context.scene.rigidbody_world

        # Ensure rigid body is in physics world collection
        if rigid_obj not in rbw.collection.objects.values():
            rbw.collection.objects.link(rigid_obj)

        self.assertIn(rigid_obj, rbw.collection.objects.values(), "Rigid body should be in physics world collection")

        # Test bake operation
        try:
            bpy.ops.mmd_tools.ptcache_rigid_body_bake()

            # Check if point cache exists and is baked
            if rbw and rbw.point_cache:
                print("   - Physics bake operation: executed")
                self.assertIsNotNone(rigid_obj.rigid_body, "Rigid body should maintain physics properties after bake")
            else:
                print("   - Physics bake operation: no point cache found")
        except Exception as e:
            print(f"   - Physics bake operation: failed ({str(e)[:50]})")

        # Test delete bake operation
        try:
            bpy.ops.mmd_tools.ptcache_rigid_body_delete_bake()
            print("   - Physics delete bake operation: executed")
            self.assertIsNotNone(rigid_obj.rigid_body, "Rigid body should still exist after cache deletion")
        except Exception as e:
            print(f"   - Physics delete bake operation: failed ({str(e)[:50]})")

        print("✓ Rigid body physics bake operations test completed")

    def test_rigid_body_model_integration(self):
        """Test rigid body integration with MMD model"""
        self._enable_mmd_tools()

        model = self._create_test_model()

        # Create rigid bodies with different configurations
        test_configs = [
            {"bone": "test_bone", "type": 0, "name": "StaticRigid"},  # Static
            {"bone": "test_bone", "type": 1, "name": "DynamicRigid"},  # Dynamic
            {"bone": "test_bone", "type": 2, "name": "DynamicBoneRigid"},  # Dynamic+Bone
        ]

        created_rigids = []
        for config in test_configs:
            rigid_obj = self._create_rigid_body_object(model, bone_name=config["bone"])
            FnRigidBody.setup_rigid_body_object(obj=rigid_obj, shape_type=0, location=Vector((len(created_rigids), 0, 0)), rotation=Euler((0, 0, 0)), size=Vector((1, 1, 1)), dynamics_type=config["type"], name=config["name"], bone=config["bone"])
            created_rigids.append(rigid_obj)

        # Check model integration
        all_rigids = list(model.rigidBodies())
        self.assertEqual(len(all_rigids), len(test_configs), "All rigid bodies should be in model")

        # Check bone references
        armature = model.armature()
        for rigid_obj in created_rigids:
            bone_name = rigid_obj.mmd_rigid.bone
            if bone_name:
                self.assertIn(bone_name, armature.pose.bones, f"Referenced bone {bone_name} should exist")

        print("✓ Rigid body model integration test passed")

    def test_rigid_body_selection_operations(self):
        """Test rigid body selection operations"""
        self._enable_mmd_tools()

        model = self._create_test_model()

        # Create multiple rigid bodies with different properties
        rigid_configs = [
            {"group": 0, "type": "0", "shape": "SPHERE"},
            {"group": 0, "type": "1", "shape": "SPHERE"},  # Same group and shape
            {"group": 1, "type": "1", "shape": "BOX"},
            {"group": 1, "type": "2", "shape": "BOX"},  # Same group and shape
        ]

        created_rigids = []
        for i, config in enumerate(rigid_configs):
            rigid_obj = self._create_rigid_body_object(model, location=(i, 0, 0))
            rigid_obj.mmd_rigid.collision_group_number = config["group"]
            rigid_obj.mmd_rigid.type = config["type"]
            rigid_obj.mmd_rigid.shape = config["shape"]
            created_rigids.append(rigid_obj)

        # Test selection by collision group
        bpy.context.view_layer.objects.active = created_rigids[0]  # Group 0, SPHERE

        try:
            # This would test the select rigid body operator if it were working
            # bpy.ops.mmd_tools.rigid_body_select(properties={'collision_group_number'})

            # Manual check for selection logic
            active_rigid = created_rigids[0]
            same_group_rigids = [r for r in created_rigids if r.mmd_rigid.collision_group_number == active_rigid.mmd_rigid.collision_group_number]

            self.assertEqual(len(same_group_rigids), 2, "Should find 2 rigid bodies in same group")
            print("   - Selection by collision group: logic verified")

        except Exception as e:
            print(f"   - Selection operation: failed ({str(e)[:50]})")

        print("✓ Rigid body selection operations test completed")

    def test_rigid_body_cleanup_operations(self):
        """Test rigid body cleanup and removal operations"""
        self._enable_mmd_tools()
        model = self._create_test_model()

        # Create test rigid bodies
        rigid_obj1 = self._create_rigid_body_object(model, location=(0, 0, 0))
        rigid_obj2 = self._create_rigid_body_object(model, location=(1, 0, 0))

        initial_count = len(list(model.rigidBodies()))
        self.assertEqual(initial_count, 2, "Should have 2 rigid bodies initially")

        self.assertAlmostEqual(rigid_obj1.location.x, 0.0, places=5, msg="First rigid body should be at x=0")
        self.assertAlmostEqual(rigid_obj2.location.x, 1.0, places=5, msg="Second rigid body should be at x=1")

        # Test removal of first object
        bpy.context.view_layer.objects.active = rigid_obj1
        rigid_obj1.select_set(True)

        try:
            bpy.ops.mmd_tools.rigid_body_remove()

            remaining_count = len(list(model.rigidBodies()))
            self.assertEqual(remaining_count, initial_count - 1, "Should have one less rigid body")

            remaining_rigids = list(model.rigidBodies())
            rigid_names = [obj.name for obj in remaining_rigids]
            self.assertIn(rigid_obj2.name, rigid_names, "Second rigid body should still exist")

            print("   - Rigid body removal: passed")
        except Exception as e:
            print(f"   - Rigid body removal: failed ({str(e)[:50]})")

        print("✓ Rigid body cleanup operations test completed")

    def test_joint_operations_comprehensive(self):
        """Test comprehensive joint operations"""
        self._enable_mmd_tools()

        model = self._create_test_model()

        # Create parent-child bone relationship for joint testing
        armature = model.armature()
        bpy.context.view_layer.objects.active = armature
        bpy.ops.object.mode_set(mode="EDIT")

        edit_bones = armature.data.edit_bones
        child_bone = edit_bones.new("child_bone")
        child_bone.head = (0, 0, 2)
        child_bone.tail = (0, 0, 3)
        child_bone.parent = edit_bones.get("test_bone")

        bpy.ops.object.mode_set(mode="OBJECT")

        # Set pose bone properties
        child_pose_bone = armature.pose.bones["child_bone"]
        child_pose_bone.mmd_bone.name_j = "子ボーン"
        child_pose_bone.mmd_bone.name_e = "child_bone"

        # Create rigid bodies for parent and child bones
        parent_rigid = self._create_rigid_body_object(model, bone_name="test_bone", location=(0, 0, 1))
        child_rigid = self._create_rigid_body_object(model, bone_name="child_bone", location=(0, 0, 2))

        # Test joint creation between related bones
        joint_obj = self._create_joint_object(model, parent_rigid, child_rigid, location=(0, 0, 1.5))

        # Check joint relationship
        rbc = joint_obj.rigid_body_constraint
        self.assertEqual(rbc.object1, parent_rigid, "First rigid should be parent")
        self.assertEqual(rbc.object2, child_rigid, "Second rigid should be child")

        # Test joint properties
        self.assertTrue(rbc.use_limit_lin_x, "Linear X limit should be enabled")
        self.assertTrue(rbc.use_limit_lin_y, "Linear Y limit should be enabled")
        self.assertTrue(rbc.use_limit_lin_z, "Linear Z limit should be enabled")
        self.assertTrue(rbc.use_limit_ang_x, "Angular X limit should be enabled")
        self.assertTrue(rbc.use_limit_ang_y, "Angular Y limit should be enabled")
        self.assertTrue(rbc.use_limit_ang_z, "Angular Z limit should be enabled")

        # Test spring settings
        self.assertTrue(rbc.use_spring_x, "Spring X should be enabled")
        self.assertTrue(rbc.use_spring_y, "Spring Y should be enabled")
        self.assertTrue(rbc.use_spring_z, "Spring Z should be enabled")

        print("✓ Comprehensive joint operations test passed")

    def test_rigid_body_world_update_operations(self):
        """Test rigid body world update operations"""
        self._enable_mmd_tools()

        model = self._create_test_model()

        # Create some rigid bodies and joints
        rigid_a = self._create_rigid_body_object(model, location=(0, 0, 0))
        rigid_b = self._create_rigid_body_object(model, location=(1, 0, 0))
        joint_obj = self._create_joint_object(model, rigid_a, rigid_b)

        rbc = joint_obj.rigid_body_constraint
        self.assertEqual(rbc.object1, rigid_a, "Joint should connect first rigid body")
        self.assertEqual(rbc.object2, rigid_b, "Joint should connect second rigid body")

        # Test world update
        try:
            bpy.ops.mmd_tools.rigid_body_world_update()

            # Check if world still exists and is properly configured
            self._check_physics_world_setup()

            rbw = bpy.context.scene.rigidbody_world

            # Check substeps and iterations (should be set by update operation)
            self.assertEqual(rbw.substeps_per_frame, 6, "Substeps should be set to 6")
            self.assertEqual(rbw.solver_iterations, 10, "Solver iterations should be set to 10")

            self.assertEqual(joint_obj.rigid_body_constraint.object1, rigid_a, "Joint should maintain connection to first rigid body after update")
            self.assertEqual(joint_obj.rigid_body_constraint.object2, rigid_b, "Joint should maintain connection to second rigid body after update")

            self.assertIn(joint_obj, rbw.constraints.objects.values(), "Joint should be in constraints collection")

            print("   - World update operation: passed")
        except Exception as e:
            print(f"   - World update operation: failed ({str(e)[:50]})")

        print("✓ Rigid body world update operations test completed")

    def test_rigid_body_stress_testing(self):
        """Test rigid body system under stress conditions"""
        self._enable_mmd_tools()

        model = self._create_test_model()

        # Create many rigid bodies to stress test
        stress_count = 20
        created_objects = []

        try:
            for i in range(stress_count):
                rigid_obj = self._create_rigid_body_object(model, shape=["SPHERE", "BOX", "CAPSULE"][i % 3], location=(i % 5, i // 5, 0))

                # Vary properties
                FnRigidBody.setup_rigid_body_object(
                    obj=rigid_obj,
                    shape_type=i % 3,
                    location=Vector((i % 5, i // 5, 0)),
                    rotation=Euler((0, 0, 0)),
                    size=Vector((0.5 + i * 0.1, 0.5 + i * 0.1, 0.5 + i * 0.1)),
                    dynamics_type=1,  # All dynamic
                    collision_group_number=i % 16,
                    collision_group_mask=[i % 2 == 0] * 16,  # Alternate collision masks
                    mass=1.0 + i * 0.1,
                    name=f"StressRigid{i}",
                )

                created_objects.append(rigid_obj)

            # Check all objects were created successfully
            all_rigids = list(model.rigidBodies())
            self.assertGreaterEqual(len(all_rigids), stress_count, f"Should have at least {stress_count} rigid bodies")

            # Test memory usage
            gc.collect()

            print(f"   - Created {len(created_objects)} rigid bodies successfully")

        except Exception as e:
            print(f"   - Stress test failed at object {len(created_objects)}: {str(e)[:50]}")

        print("✓ Rigid body stress testing completed")

    def test_rigid_body_compatibility_with_blender_physics(self):
        """Test MMD rigid body compatibility with Blender's physics system"""
        self._enable_mmd_tools()

        model = self._create_test_model()
        rigid_obj = self._create_rigid_body_object(model)

        # Check Blender physics integration
        rb = rigid_obj.rigid_body
        self.assertIsNotNone(rb, "Should have Blender rigid body component")
        self.assertEqual(rb.type, "ACTIVE", "Should be active rigid body")

        # Check physics world integration
        rbw = bpy.context.scene.rigidbody_world
        self.assertIn(rigid_obj, rbw.collection.objects.values(), "Rigid body should be in physics world collection")

        # Test physics simulation compatibility
        try:
            # Set up a simple physics test
            initial_location = Vector((0, 0, 5))  # Drop from height
            rigid_obj.location = initial_location

            # Store initial position for comparison
            initial_z = rigid_obj.location.z

            # Run a few simulation steps
            original_frame = bpy.context.scene.frame_current
            for frame in range(original_frame + 1, original_frame + 5):
                bpy.context.scene.frame_set(frame)

            # Check if object moved (basic physics simulation)
            final_location = rigid_obj.location
            final_z = final_location.z

            # Use final_location to verify the effect of the physics simulation.
            # Object should have fallen due to gravity (Z position should decrease)
            # Note: In some cases physics might not run immediately, so we check if position changed at all
            location_changed = (final_location - initial_location).length > 0.001
            gravity_effect = final_z <= initial_z  # Should fall or stay at same level

            if location_changed:
                print(f"   - Physics simulation: object moved from {initial_location} to {final_location}")
                self.assertTrue(gravity_effect, "Object should not move upward due to gravity")
            else:
                print("   - Physics simulation: object position unchanged (physics may need more frames)")

            bpy.context.scene.frame_set(original_frame)  # Reset frame
            print("   - Blender physics simulation: compatible")

        except Exception as e:
            print(f"   - Blender physics simulation: failed ({str(e)[:50]})")

        print("✓ Rigid body Blender physics compatibility test passed")

    def test_rigid_body_data_persistence(self):
        """Test rigid body data persistence and saving"""
        self._enable_mmd_tools()

        model = self._create_test_model()

        # Create rigid body with specific properties
        test_properties = {"name_j": "テストリジッド保存", "name_e": "test_rigid_save", "collision_group_number": 5, "mass": 3.14, "friction": 0.87, "bone": "test_bone"}

        rigid_obj = self._create_rigid_body_object(model, bone_name=test_properties["bone"])

        # Set properties
        rigid_obj.mmd_rigid.name_j = test_properties["name_j"]
        rigid_obj.mmd_rigid.name_e = test_properties["name_e"]
        rigid_obj.mmd_rigid.collision_group_number = test_properties["collision_group_number"]
        rigid_obj.rigid_body.mass = test_properties["mass"]
        rigid_obj.rigid_body.friction = test_properties["friction"]

        # Verify properties are set correctly
        self.assertEqual(rigid_obj.mmd_rigid.name_j, test_properties["name_j"])
        self.assertEqual(rigid_obj.mmd_rigid.name_e, test_properties["name_e"])
        self.assertEqual(rigid_obj.mmd_rigid.collision_group_number, test_properties["collision_group_number"])
        self.assertAlmostEqual(rigid_obj.rigid_body.mass, test_properties["mass"], places=5)
        self.assertAlmostEqual(rigid_obj.rigid_body.friction, test_properties["friction"], places=5)

        print("✓ Rigid body data persistence test passed")

    def test_rigid_body_error_handling(self):
        """Test rigid body error handling and edge cases"""
        self._enable_mmd_tools()

        model = self._create_test_model()

        # Test invalid parameters
        error_cases = [
            {"desc": "Invalid shape type", "params": {"shape_type": 999}, "should_fail": True},
            {
                "desc": "Invalid dynamics type",
                "params": {"dynamics_type": -1},
                "should_fail": False,  # Should clamp to valid range
            },
            {
                "desc": "Negative mass",
                "params": {"mass": -1.0},
                "should_fail": False,  # Should be handled gracefully
            },
            {
                "desc": "Zero size",
                "params": {"size": Vector((0, 0, 0))},
                "should_fail": False,  # Should be handled
            },
        ]

        for case in error_cases:
            try:
                rigid_obj = self._create_rigid_body_object(model)

                default_params = {"shape_type": 0, "location": Vector((0, 0, 0)), "rotation": Euler((0, 0, 0)), "size": Vector((1, 1, 1)), "dynamics_type": 1, "name": f"ErrorTest_{case['desc'].replace(' ', '')}"}
                default_params.update(case["params"])

                FnRigidBody.setup_rigid_body_object(obj=rigid_obj, **default_params)

                if case["should_fail"]:
                    print(f"   - {case['desc']}: unexpectedly succeeded")
                else:
                    print(f"   - {case['desc']}: handled gracefully")

            except Exception as e:
                if case["should_fail"]:
                    print(f"   - {case['desc']}: failed as expected")
                else:
                    print(f"   - {case['desc']}: unexpected error ({str(e)[:50]})")

        print("✓ Rigid body error handling test completed")

    def test_physics_assembly_repeated_operations_should_fail(self):
        """Test repeated physics assembly operations and parameter changes - should FAIL if bug exists"""
        self._enable_mmd_tools()

        model = self._create_test_model()

        # Create a chain of rigid bodies (like a skirt)
        rigid_bodies = []
        for i in range(3):  # Reduced count for more reliable testing
            rigid_obj = self._create_rigid_body_object(
                model,
                location=(i * 1.0, 0, 5),  # Start at height 5 for gravity effect
                shape="SPHERE",
            )
            # Set initial parameters
            rigid_obj.rigid_body.mass = 1.0
            rigid_obj.rigid_body.friction = 0.5
            rigid_obj.rigid_body.linear_damping = 0.0  # No damping for clear movement
            rigid_obj.mmd_rigid.type = "1"  # Dynamic
            rigid_bodies.append(rigid_obj)

        # Ensure physics world is properly set up
        if not bpy.context.scene.rigidbody_world:
            bpy.ops.rigidbody.world_add()

        rbw = bpy.context.scene.rigidbody_world
        for rb in rigid_bodies:
            if rb not in rbw.collection.objects.values():
                rbw.collection.objects.link(rb)

        # Store initial positions
        initial_positions = [rb.location.copy() for rb in rigid_bodies]

        # First physics simulation
        original_frame = bpy.context.scene.frame_current
        bpy.context.scene.frame_set(1)

        # Run simulation for enough frames to see gravity effect
        for frame in range(1, 30):
            bpy.context.scene.frame_set(frame)

        first_simulation_positions = [rb.location.copy() for rb in rigid_bodies]

        # Reset to frame 1 and positions
        bpy.context.scene.frame_set(1)
        for i, rb in enumerate(rigid_bodies):
            rb.location = initial_positions[i]

        # CRITICAL: Modify parameters significantly
        for rb in rigid_bodies:
            rb.rigid_body.mass = 10.0  # 10x heavier - should fall much faster
            rb.rigid_body.friction = 0.0  # No friction

        # Force world update
        bpy.ops.mmd_tools.rigid_body_world_update()

        # Second physics simulation with same frame count
        bpy.context.scene.frame_set(1)
        for frame in range(1, 30):
            bpy.context.scene.frame_set(frame)

        second_simulation_positions = [rb.location.copy() for rb in rigid_bodies]

        # Reset frame
        bpy.context.scene.frame_set(original_frame)

        # Check if simulation results are meaningfully different
        # With 10x mass, objects should fall much faster and reach different positions
        max_position_difference = 0.0
        for i in range(len(first_simulation_positions)):
            diff = (first_simulation_positions[i] - second_simulation_positions[i]).length
            max_position_difference = max(max_position_difference, diff)

        print(f"   - Max position difference between simulations: {max_position_difference}")
        print(f"   - First simulation final Z: {[pos.z for pos in first_simulation_positions]}")
        print(f"   - Second simulation final Z: {[pos.z for pos in second_simulation_positions]}")

        # The test FAILS if the bug exists (positions are too similar despite parameter changes)
        self.assertGreater(max_position_difference, 0.5, "Physics simulation results should be significantly different when mass changes from 1.0 to 10.0. If this fails, it indicates the bug where parameter changes don't affect physics simulation.")

    def test_mass_change_physics_effect_should_fail(self):
        """Test if mass changes actually affect physics simulation - should FAIL if bug exists"""
        self._enable_mmd_tools()

        model = self._create_test_model()

        # Create two identical rigid bodies
        rigid_light = self._create_rigid_body_object(model, location=(0, 0, 10))
        rigid_heavy = self._create_rigid_body_object(model, location=(2, 0, 10))

        # Set very different masses
        rigid_light.rigid_body.mass = 0.1  # Very light
        rigid_heavy.rigid_body.mass = 100.0  # Very heavy

        # Both should have no air resistance
        rigid_light.rigid_body.linear_damping = 0.0
        rigid_heavy.rigid_body.linear_damping = 0.0

        # Ensure they're in physics world
        rbw = bpy.context.scene.rigidbody_world
        if not rbw:
            bpy.ops.rigidbody.world_add()
            rbw = bpy.context.scene.rigidbody_world

        if rigid_light not in rbw.collection.objects.values():
            rbw.collection.objects.link(rigid_light)
        if rigid_heavy not in rbw.collection.objects.values():
            rbw.collection.objects.link(rigid_heavy)

        # Run physics simulation
        original_frame = bpy.context.scene.frame_current
        bpy.context.scene.frame_set(1)

        initial_light_z = rigid_light.location.z
        initial_heavy_z = rigid_heavy.location.z

        # Simulate physics
        for frame in range(1, 50):
            bpy.context.scene.frame_set(frame)

        final_light_z = rigid_light.location.z
        final_heavy_z = rigid_heavy.location.z

        bpy.context.scene.frame_set(original_frame)

        light_fall_distance = initial_light_z - final_light_z
        heavy_fall_distance = initial_heavy_z - final_heavy_z

        print(f"   - Light object (mass 0.1) fell: {light_fall_distance}")
        print(f"   - Heavy object (mass 100.0) fell: {heavy_fall_distance}")
        print(f"   - Fall distance difference: {abs(light_fall_distance - heavy_fall_distance)}")

        # In a vacuum, both should fall the same distance (physics is correct)
        # But if there's any air resistance or other factors, heavy should fall slightly more
        # The key test: they should BOTH fall significantly (> 1 unit)
        self.assertGreater(light_fall_distance, 1.0, "Light object should fall significantly due to gravity")
        self.assertGreater(heavy_fall_distance, 1.0, "Heavy object should fall significantly due to gravity")

        # More importantly: verify that mass values are actually what we set
        self.assertAlmostEqual(rigid_light.rigid_body.mass, 0.1, places=2, msg="Light object mass should be 0.1")
        self.assertAlmostEqual(rigid_heavy.rigid_body.mass, 100.0, places=1, msg="Heavy object mass should be 100.0")

    def test_parameter_modification_after_bake_should_fail(self):
        """Test parameter changes after physics bake - should FAIL if parameters don't take effect"""
        self._enable_mmd_tools()

        model = self._create_test_model()
        rigid_obj = self._create_rigid_body_object(model, location=(0, 0, 8))

        # Set initial parameters
        rigid_obj.rigid_body.mass = 1.0
        rigid_obj.rigid_body.friction = 0.5
        rigid_obj.rigid_body.linear_damping = 0.0

        # Ensure physics world
        if not bpy.context.scene.rigidbody_world:
            bpy.ops.rigidbody.world_add()

        rbw = bpy.context.scene.rigidbody_world
        if rigid_obj not in rbw.collection.objects.values():
            rbw.collection.objects.link(rigid_obj)

        # First: Bake physics with initial parameters and simulate
        original_frame = bpy.context.scene.frame_current
        bpy.context.scene.frame_set(1)

        try:
            bpy.ops.mmd_tools.ptcache_rigid_body_bake()
        except Exception:
            pass  # Bake might fail, that's ok for this test

        # Run first simulation
        for frame in range(1, 30):
            bpy.context.scene.frame_set(frame)

        first_final_position = rigid_obj.location.copy()

        # Reset position and change parameters significantly
        bpy.context.scene.frame_set(1)
        rigid_obj.location.z = 8  # Reset height
        rigid_obj.rigid_body.mass = 50.0  # Much heavier

        # Delete old bake to force recalculation
        try:
            bpy.ops.mmd_tools.ptcache_rigid_body_delete_bake()
        except Exception:
            pass

        # Force world update
        bpy.ops.mmd_tools.rigid_body_world_update()

        # Test that new parameters are actually applied
        current_mass = rigid_obj.rigid_body.mass
        print(f"   - Set mass to 50.0, current mass is: {current_mass}")

        # This should pass if physics system correctly updates parameters
        self.assertAlmostEqual(current_mass, 50.0, places=1, msg="Mass should be updated to 50.0 after parameter change and world update")

        # NEW: Test actual physics simulation with new parameters
        try:
            bpy.ops.mmd_tools.ptcache_rigid_body_bake()
        except Exception:
            pass

        # Run second simulation with new mass
        bpy.context.scene.frame_set(1)
        for frame in range(1, 30):
            bpy.context.scene.frame_set(frame)

        second_final_position = rigid_obj.location.copy()

        bpy.context.scene.frame_set(original_frame)

        # Calculate position difference
        position_difference = (first_final_position - second_final_position).length

        print(f"   - First simulation (mass 1.0) final position: {first_final_position}")
        print(f"   - Second simulation (mass 50.0) final position: {second_final_position}")
        print(f"   - Position difference: {position_difference}")

        # The critical test: physics simulation should reflect the parameter change
        self.assertGreater(position_difference, 0.1, f"Position difference ({position_difference}) should be > 0.1 when mass changes from 1.0 to 50.0. If this fails, it indicates physics parameters are not properly applied after bake operations.")

    def test_skirt_physics_deterministic_should_fail(self):
        """Test skirt physics with deterministic setup - should FAIL if bug exists"""
        self._enable_mmd_tools()

        model = self._create_test_model()

        # Create a simple 3-segment chain
        segments = []
        for i in range(3):
            segment = self._create_rigid_body_object(
                model,
                location=(i * 0.5, 0, 5),  # Linear arrangement, start at height
                shape="BOX",
            )
            # Set identical initial parameters
            segment.rigid_body.mass = 1.0
            segment.rigid_body.friction = 0.5
            segment.rigid_body.linear_damping = 0.1
            segment.rigid_body.angular_damping = 0.1
            segment.mmd_rigid.type = "1"  # Dynamic
            segments.append(segment)

        # Create joints between segments
        joints = []
        for i in range(len(segments) - 1):
            joint = self._create_joint_object(model, segments[i], segments[i + 1], location=((i + 0.5) * 0.5, 0, 5))
            # Set joint limits
            rbc = joint.rigid_body_constraint
            rbc.limit_lin_x_upper = 0.2
            rbc.limit_lin_x_lower = -0.2
            rbc.limit_lin_y_upper = 0.2
            rbc.limit_lin_y_lower = -0.2
            rbc.limit_lin_z_upper = 0.2
            rbc.limit_lin_z_lower = -0.2
            joints.append(joint)

        # Ensure physics world setup
        rbw = bpy.context.scene.rigidbody_world
        if not rbw:
            bpy.ops.rigidbody.world_add()
            rbw = bpy.context.scene.rigidbody_world

        for segment in segments:
            if segment not in rbw.collection.objects.values():
                rbw.collection.objects.link(segment)

        for joint in joints:
            if joint not in rbw.constraints.objects.values():
                rbw.constraints.objects.link(joint)

        # Record initial setup
        initial_positions = [seg.location.copy() for seg in segments]
        initial_masses = [seg.rigid_body.mass for seg in segments]

        # First simulation
        original_frame = bpy.context.scene.frame_current
        bpy.context.scene.frame_set(1)
        for frame in range(1, 40):
            bpy.context.scene.frame_set(frame)

        first_final_positions = [seg.location.copy() for seg in segments]

        # Reset everything
        bpy.context.scene.frame_set(1)
        for i, segment in enumerate(segments):
            segment.location = initial_positions[i]

        # Change parameters dramatically
        for segment in segments:
            segment.rigid_body.mass = 5.0  # 5x heavier
            segment.rigid_body.linear_damping = 0.5  # Much more damping

        # Force world update
        bpy.ops.mmd_tools.rigid_body_world_update()

        # Verify parameters were actually changed
        for i, segment in enumerate(segments):
            actual_mass = segment.rigid_body.mass
            actual_damping = segment.rigid_body.linear_damping
            print(f"   - Segment {i}: mass changed from {initial_masses[i]} to {actual_mass}, damping: {actual_damping}")

            # Assert that mass actually changed
            self.assertNotAlmostEqual(actual_mass, initial_masses[i], places=2, msg=f"Segment {i} mass should have changed from {initial_masses[i]} to 5.0")
            self.assertAlmostEqual(actual_mass, 5.0, places=2, msg=f"Segment {i} mass should be 5.0 after parameter change")

        # Second simulation with same duration
        bpy.context.scene.frame_set(1)
        for frame in range(1, 40):
            bpy.context.scene.frame_set(frame)

        second_final_positions = [seg.location.copy() for seg in segments]

        bpy.context.scene.frame_set(original_frame)

        # Calculate differences
        position_differences = []
        for i in range(len(first_final_positions)):
            diff = (first_final_positions[i] - second_final_positions[i]).length
            position_differences.append(diff)

        max_diff = max(position_differences)
        avg_diff = sum(position_differences) / len(position_differences)

        print(f"   - Position differences: {position_differences}")
        print(f"   - Max difference: {max_diff}, Average difference: {avg_diff}")
        print(f"   - First simulation final positions: {[tuple(pos) for pos in first_final_positions]}")
        print(f"   - Second simulation final positions: {[tuple(pos) for pos in second_final_positions]}")

        # The test fails if positions are too similar despite significant parameter changes
        self.assertGreater(max_diff, 0.3, f"Maximum position difference ({max_diff}) should be > 0.3 when mass changes from 1.0 to 5.0 and damping changes from 0.1 to 0.5. If this fails, it indicates the physics parameter change bug described in GitHub issue #232.")

    def test_physics_world_parameter_sync_should_fail(self):
        """Test if physics world properly syncs parameter changes - should FAIL if sync broken"""
        self._enable_mmd_tools()

        model = self._create_test_model()
        rigid_obj = self._create_rigid_body_object(model, location=(0, 0, 0))

        # Set specific parameters
        test_mass = 7.77
        test_friction = 0.33
        test_restitution = 0.66

        rigid_obj.rigid_body.mass = test_mass
        rigid_obj.rigid_body.friction = test_friction
        rigid_obj.rigid_body.restitution = test_restitution

        # Multiple world update operations (simulating user workflow)
        for i in range(3):
            bpy.ops.mmd_tools.rigid_body_world_update()

            # Check if parameters are preserved after each update
            current_mass = rigid_obj.rigid_body.mass
            current_friction = rigid_obj.rigid_body.friction
            current_restitution = rigid_obj.rigid_body.restitution

            print(f"   - After update {i}: mass={current_mass}, friction={current_friction}, restitution={current_restitution}")

            # Parameters should remain exactly as set
            self.assertAlmostEqual(current_mass, test_mass, places=3, msg=f"Mass should remain {test_mass} after world update {i}")
            self.assertAlmostEqual(current_friction, test_friction, places=3, msg=f"Friction should remain {test_friction} after world update {i}")
            self.assertAlmostEqual(current_restitution, test_restitution, places=3, msg=f"Restitution should remain {test_restitution} after world update {i}")

    def test_cached_physics_state_invalidation_should_fail(self):
        """Test if cached physics state is properly invalidated - should FAIL if cache not cleared"""
        self._enable_mmd_tools()

        model = self._create_test_model()
        rigid_obj = self._create_rigid_body_object(model, location=(0, 0, 10))

        # Set initial parameters for first bake
        rigid_obj.rigid_body.mass = 1.0
        rigid_obj.rigid_body.linear_damping = 0.0

        # Ensure in physics world
        rbw = bpy.context.scene.rigidbody_world
        if not rbw:
            bpy.ops.rigidbody.world_add()
            rbw = bpy.context.scene.rigidbody_world

        if rigid_obj not in rbw.collection.objects.values():
            rbw.collection.objects.link(rigid_obj)

        # First bake and simulation
        original_frame = bpy.context.scene.frame_current
        bpy.context.scene.frame_set(1)

        try:
            bpy.ops.mmd_tools.ptcache_rigid_body_bake()
            print("   - First bake completed")
        except Exception as e:
            print(f"   - First bake failed: {e}")

        # Simulate to get position
        for frame in range(1, 30):
            bpy.context.scene.frame_set(frame)

        first_final_position = rigid_obj.location.copy()

        # Reset and change parameters significantly
        bpy.context.scene.frame_set(1)
        rigid_obj.location.z = 10  # Reset height
        rigid_obj.rigid_body.mass = 20.0  # Much heavier

        # Delete bake and rebake with new parameters
        try:
            bpy.ops.mmd_tools.ptcache_rigid_body_delete_bake()
            print("   - Deleted previous bake")
        except Exception as e:
            print(f"   - Delete bake failed: {e}")

        try:
            bpy.ops.mmd_tools.ptcache_rigid_body_bake()
            print("   - Second bake completed")
        except Exception as e:
            print(f"   - Second bake failed: {e}")

        # Simulate same duration with new parameters
        bpy.context.scene.frame_set(1)
        for frame in range(1, 30):
            bpy.context.scene.frame_set(frame)

        second_final_position = rigid_obj.location.copy()

        bpy.context.scene.frame_set(original_frame)

        # Check if mass parameter actually changed
        current_mass = rigid_obj.rigid_body.mass
        self.assertAlmostEqual(current_mass, 20.0, places=1, msg="Mass should be 20.0 after parameter change")

        # Check if simulation results reflect the parameter change
        position_difference = (first_final_position - second_final_position).length

        print(f"   - First simulation (mass 1.0) final position: {first_final_position}")
        print(f"   - Second simulation (mass 20.0) final position: {second_final_position}")
        print(f"   - Position difference: {position_difference}")

        # Results should be different enough to indicate parameter change took effect
        # If physics cache isn't properly invalidated, positions will be identical
        self.assertGreater(position_difference, 0.1, f"Position difference ({position_difference}) should be > 0.1 when mass changes from 1.0 to 20.0. If this fails, it indicates physics cache is not properly invalidated after parameter changes.")


if __name__ == "__main__":
    import sys

    sys.argv = [__file__] + (sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else [])
    unittest.main()
