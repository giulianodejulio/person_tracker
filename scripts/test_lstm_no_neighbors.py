from . import trajnet_evaluator as te
import argparse
import trajnetplusplustools

# Convert lists to numpy arrays for easier manipulation (optional)
import numpy as np
import matplotlib.pyplot as plt

import json
import os


def add_arguments(parser):
    parser.add_argument('--path', default='DATA_BLOCK/synth_data/test_pred/',
                        help='directory of data to test')
    parser.add_argument('--output', nargs='+',
                        help='relative path to saved model')
    parser.add_argument('--obs_length', default=9, type=int,
                        help='observation length')
    parser.add_argument('--pred_length', default=12, type=int,
                        help='prediction length')
    parser.add_argument('--write_only', action='store_true',
                        help='disable writing new files')
    parser.add_argument('--disable-collision', action='store_true',
                        help='disable collision metrics')
    parser.add_argument('--labels', required=False, nargs='+',
                        help='labels of models')
    parser.add_argument('--normalize_scene', action='store_true',
                        help='augment scenes')
    parser.add_argument('--modes', default=1, type=int,
                        help='number of modes to predict')
    parser.add_argument('--scene_id', default=1888, type=int,
                        help='scene id')
    return parser.parse_args()


def write_no_neighbors_json(input_file, no_neighbors_file, scene_id):
    # Read the input NDJSON file
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    # Parse the NDJSON file
    scenes = []
    tracks = []
    for line in lines:
        data = json.loads(line)
        if "scene" in data:
            scenes.append(data)
        elif "track" in data:
            tracks.append(data)
    
    # Find the specific scene
    target_scene = None
    for scene in scenes:
        if scene["scene"]["id"] == scene_id:
            target_scene = scene
            break
    
    if not target_scene:
        raise ValueError(f"Scene with id {scene_id} not found.")
    
    # Find the corresponding tracks
    scene_p = target_scene["scene"]["p"]
    start_frame = target_scene["scene"]["s"]
    end_frame = target_scene["scene"]["e"]
    
    # Collect tracks for the target pedestrian and neighboring pedestrians
    target_tracks = []
    neighbor_tracks = {p: [] for p in range(scene_p - 2, scene_p + 3) if p != scene_p}
    
    for track in tracks:
        if track["track"]["f"] >= start_frame and track["track"]["f"] <= end_frame:
            if track["track"]["p"] == scene_p:
                target_tracks.append(track)
            elif track["track"]["p"] in neighbor_tracks:
                neighbor_tracks[track["track"]["p"]].append(track)
    
    # Prepare the output data
    output_data = [target_scene]
    for track in target_tracks:
        output_data.append(track)
    
    # for neighbor_p, tracks in neighbor_tracks.items():
    #     for track in tracks:
    #         track["track"]["x"] = 0.0
    #         track["track"]["y"] = 0.0
    #         output_data.append(track)

    # Write to the output NDJSON file
    with open(no_neighbors_file, 'w') as f:
        for item in output_data:
            f.write(json.dumps(item) + "\n")


def remove_first_row_element(scenes):
    # Create a new list to hold the modified scenes
    modified_scenes = []

    for scene in scenes:
        path, scene_id, row_list = scene
        # Remove the first Row element from the row_list
        if row_list and row_list[0]:
            row_list[0] = row_list[0][1:]
        # Append the modified scene to the new list
        modified_scenes.append((path, scene_id, row_list))
    
    return modified_scenes


def read_dataset_from_test_file(no_neighbors_file, scene_ids_list):
    test_dataset = no_neighbors_file.replace('.ndjson', '')         # this has only the rows of a single scene (e.g. scene 1888) and no neighbors' trajectories
    # test_dataset = 'DATA_BLOCK/synth_data/test/orca_five_synth'   # Complete test Dataset with all the scenes and trajectories.
    ground_truth_dataset = 'DATA_BLOCK/synth_data/test_private/orca_five_synth'

    reader      = trajnetplusplustools.Reader(test_dataset + '.ndjson', scene_type='paths')
    scenes      = [(test_dataset, s_id, s) for s_id, s in reader.scenes(ids=scene_ids_list)]
    scene_goals = [np.zeros((len(paths), 2)) for _, scene_id, paths in scenes]

    reader_ground_truth = trajnetplusplustools.Reader(ground_truth_dataset + '.ndjson', scene_type='paths')
    scenes_ground_truth = [(ground_truth_dataset, s_id, s) for s_id, s in reader_ground_truth.scenes(ids=scene_ids_list)]

    return scenes, scene_goals, scenes_ground_truth


def extract_coordinates(scenes, predictions):
    # Extract the observed trajectory for the primary pedestrian
    obs_trajectory = scenes[0][2][0]  # Primary pedestrian's data in the scene
    obs_traj = [[row.x, row.y] for row in obs_trajectory]
    
    # Extract the predicted trajectory for the primary pedestrian (first array in predictions)
    pred_trajectory = predictions[0][0]  # First array in predictions
    pred_traj = pred_trajectory.tolist()
    
    return obs_traj, pred_traj


def compute_plot_limits(obs_traj, pred_traj, ground_truth_traj):
    #Combine all points to determine the limits
    all_points = np.vstack((obs_traj, pred_traj, ground_truth_traj))
    # Get the min and max for x and y
    min_x, max_x = all_points[:, 0].min(), all_points[:, 0].max()
    min_y, max_y = all_points[:, 1].min(), all_points[:, 1].max()
    # Add some padding to the limits
    padding = 0.5
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding
    return min_x, max_x, min_y, max_y


def plot_coordinates(observed_trajectory, predicted_trajectory, ground_truth_trajectory, scene_ids_list, args, idx, min_x, max_x, min_y, max_y):
    observed_trajectory     = np.array(observed_trajectory)
    predicted_trajectory    = np.array(predicted_trajectory)
    ground_truth_trajectory = np.array(ground_truth_trajectory)

    # Plot the trajectories
    plt.figure(figsize=(10, 6))

    plt.plot(ground_truth_trajectory[:, 0], ground_truth_trajectory[:, 1], 'k--o', label='test_private/orca_five_synth (Ground Truth Trajectory)')
    plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], 'r-o', label='test_pred/orca_five_synth (Predicted Trajectory)')
    plt.plot(observed_trajectory[0:len(observed_trajectory), 0],
             observed_trajectory[0:len(observed_trajectory), 1],
             'g-o', label='test/orca_five_synth (actually no_neighbors.ndjson Trajectory)')
    plt.plot(observed_trajectory[0:args.obs_length, 0],
             observed_trajectory[0:args.obs_length, 1],
             'b-o', label='Observed Trajectory')

    # plt.scatter(predicted_trajectory[0, 0], predicted_trajectory[0, 1], color='red', zorder=5, s=100, marker='o')
    plt.scatter(observed_trajectory[0, 0], observed_trajectory[0, 1], color='magenta', zorder=5, s=100, marker='o')
    plt.scatter(observed_trajectory[args.obs_length-1, 0], observed_trajectory[args.obs_length-1, 1], color='magenta', zorder=5, s=100, marker='o')

    # Set the limits
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    # Add labels and title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Trajectories for Primary Pedestrian in Scene ' + str(scene_ids_list[0]))
    plt.legend()

    # Show grid
    plt.grid(True)

    # Show the plot
    # plt.show()

    # Define the directory to save the plot
    save_dir = f"trajnetbaselines/lstm/visualizations/{scene_ids_list[0]}"
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it does not exist

    # Save the plot as a PNG file
    file_name = f"{idx}.png"
    file_path = os.path.join(save_dir, file_name)
    plt.savefig(file_path)

    # Close the plot to free memory
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    args = add_arguments(parser)

    model      = "OUTPUT_BLOCK/synth_data/lstm_vanilla_None.pkl"
    predictor  = te.load_predictor(model)

    ## no_neighbors.ndjson file generation
    scene_id    = args.scene_id
    input_file  = 'DATA_BLOCK/synth_data/test/orca_five_synth.ndjson'
    no_neighbors_file = 'DATA_BLOCK/synth_data/test/no_neighbors.ndjson'
    write_no_neighbors_json(input_file, no_neighbors_file, scene_id)

    scene_ids_list = [scene_id]
    scenes, scene_goals, scenes_ground_truth = read_dataset_from_test_file(no_neighbors_file, scene_ids_list=scene_ids_list)
    # print("scenes: ", scenes)
    ## load a moving window in the trajectory
    idx = 0
    test_traj_length = len(scenes[0][2][0]) # length of trajectory which will be explored by the observed trajectory
    while test_traj_length >= args.obs_length:
        if idx > 0:
            scenes = remove_first_row_element(scenes)
        for (_, _, paths), scene_goal in zip(scenes, scene_goals):
            print("paths: ",paths)
            predictions = predictor(paths, scene_goal, n_predict=12, obs_length=9, modes=1, args=args)
        # te.write_predictions(pred_list, scenes, model_name, dataset_name, args)
        
        obs_traj, pred_traj  = extract_coordinates(scenes, predictions)
        ground_truth_traj, _ = extract_coordinates(scenes_ground_truth, predictions)

        if idx == 0:
            min_x, max_x, min_y, max_y = compute_plot_limits(obs_traj, pred_traj, ground_truth_traj)
        plot_coordinates(obs_traj, pred_traj, ground_truth_traj, scene_ids_list, args, idx, min_x, max_x, min_y, max_y)
        idx += 1
        test_traj_length -= 1


if __name__ == '__main__':
    main()
