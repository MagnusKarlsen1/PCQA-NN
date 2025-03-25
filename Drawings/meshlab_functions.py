import pymeshlab


def sample_stl_by_point_distance(input_stl, output_xyz, sampling_distance):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(input_stl)


    # Perform Poisson Disk sampling (or Monte Carlo) to get uniform points
    ms.generate_sampling_poisson_disk(
        radius=pymeshlab.PureValue(sampling_distance)

    )
   
    # Save as XYZ point cloud (you can also save as PLY, STL, etc.)
    ms.save_current_mesh(output_xyz)

    print(f"✅ Resampled mesh saved to: {output_xyz}")


































