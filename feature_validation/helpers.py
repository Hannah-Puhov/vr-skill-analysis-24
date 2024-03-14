import numpy as np
from sklearn.metrics import mean_squared_error
import h5py
import feature_extraction as ft

def open_file(file):
        f = h5py.File(file, 'r')

        data = f['data']
        v_rm = f['voxels_removed']
        force = []
        if 'drill_force_feedback' in f:
            force = f['drill_force_feedback']
        elif 'force' in f:
            force = f['force']

        return data, force, v_rm

def get_mean_abs_velocity(file, eval_metrics):
        data, _, _ = open_file(file)
        strokes, _ = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

        inds = ft.get_stroke_indices(strokes)

        velocities, _ = ft.extract_kinematics(
            data['pose_mastoidectomy_drill'][()], data['time'][()], inds)

        mean, _, _, _ = ft.stats_per_stroke(make_pos(velocities))
        eval_metrics.kinematics.velocity.add_mean(mean)
        return mean

def get_mean_abs_acceleration(file, eval_metrics):
        data, _, _ = open_file(file)
        strokes, _ = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

        inds = ft.get_stroke_indices(strokes)

        _, accelerations = ft.extract_kinematics(
            data['pose_mastoidectomy_drill'][()], data['time'][()], inds)

        mean, _, _, _ = ft.stats_per_stroke(make_pos(accelerations))
        eval_metrics.kinematics.acceleration.add_mean(mean)
        return mean

def get_mean_abs_jerk(file, eval_metrics):
        data, _, _ = open_file(file)
        strokes, _ = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

        inds = ft.get_stroke_indices(strokes)

        jerks = ft.extract_jerk(
            data['pose_mastoidectomy_drill'][()], data['time'][()], inds)

        mean, _, _, _ = ft.stats_per_stroke(make_pos(jerks))
        eval_metrics.kinematics.jerk.add_mean(mean)
        return mean

def get_max_stroke_force(file, eval_metrics):
    data, force, _ = open_file(file)


    strokes, stroke_times = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

    mean, _, maxi, _ = ft.stats_per_stroke(ft.stroke_force(
        strokes, stroke_times, force['wrench'][()], force['time_stamp'][()]))
    eval_metrics.strokes.force.add_mean(mean)

    return maxi

def get_mean_stroke_length(file, eval_metrics):
    data, _, _ = open_file(file)
    strokes, _ = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

    mean, _, _, _ = ft.stats_per_stroke(ft.stroke_length(
        np.array(strokes), data['pose_mastoidectomy_drill'][()]))
    eval_metrics.strokes.length.add_mean(mean)

    return mean

def make_pos(arr):
    arr = [abs(val) for val in arr]
    return np.array(arr)

def get_mean_curvature(file, eval_metrics):
      
    data, _, _ = open_file(file)

    strokes, _ = ft.get_strokes(
    data['pose_mastoidectomy_drill'][()], data['time'][()])

    inds = ft.get_stroke_indices(strokes)

    mean, _, _, _ = ft.stats_per_stroke(ft.extract_curvature(
        data['pose_mastoidectomy_drill'][()], data['time'][()], inds))
    eval_metrics.strokes.curvature.add_mean(mean)

    return mean

def get_mean_angle(file):
    data, force, _ = open_file(file)

    strokes, stroke_times = ft.get_strokes(
    data['pose_mastoidectomy_drill'][()], data['time'][()])

    mean, _, _, _ = ft.stats_per_stroke(ft.drill_orientation(
        strokes, stroke_times, data['pose_mastoidectomy_drill'][()], data['time'][()], force['wrench'][()], force['time_stamp'][()]*1e9))
    
    return mean

def get_mean_removal_rate(file, eval_metrics):
    data, _, v_rm = open_file(file)


    strokes, stroke_times = ft.get_strokes(
        data['pose_mastoidectomy_drill'][()], data['time'][()])

    mean, _, _, _ = ft.stats_per_stroke(ft.bone_removal_rate(
        strokes, stroke_times, data['pose_mastoidectomy_drill'][()], v_rm['voxel_time_stamp'][()]))
    eval_metrics.removal_rate.add_mean(mean)

    return mean

