import unittest
import os
import numpy as np
import feature_extraction as ft
from rich.progress import track
from evaluation_metrics import EvaluationMetrics
from helpers import (open_file, get_mean_abs_acceleration, get_mean_abs_velocity,
                     get_mean_abs_jerk, get_max_stroke_force, get_mean_stroke_length,
                     get_mean_curvature, get_mean_angle, get_mean_removal_rate)

class TestValidation(unittest.TestCase):

    eval_metrics = EvaluationMetrics()
        
    def test_zero_stroke_count(self):
        file = 'Strokes/zero_strokes.hdf5'
        data, _, _ = open_file(file)
        strokes, _ = ft.get_strokes(data['pose_mastoidectomy_drill'][()], data['time'][()])
        self.assertEqual(sum(strokes), 0, f'Expected 0 strokes in {file}')

    def test_three_stroke_count(self):
        file = 'Strokes/three_strokes.hdf5'
        data, _, _ = open_file(file)
        strokes, _ = ft.get_strokes(data['pose_mastoidectomy_drill'][()], data['time'][()])
        self.assertEqual(sum(strokes), 3, f'Expected 0 strokes in {file}')

    def test_many_stroke_count(self):
        file = 'Strokes/many_strokes.hdf5'
        data, _, _ = open_file(file)
        strokes, _ = ft.get_strokes(data['pose_mastoidectomy_drill'][()], data['time'][()])
        self.assertGreater(sum(strokes), 3, f'Expected 0 strokes in {file}')

    def test_kinematics_constant_velocity(self):
        slowFile = 'Kinematics/slow_constant.hdf5'
        fastFile = 'Kinematics/fast_constant.hdf5'
        slowVelocity = get_mean_abs_velocity(slowFile, self.eval_metrics)
        fastVelocity = get_mean_abs_velocity(fastFile, self.eval_metrics)
        self.assertLess(slowVelocity, fastVelocity,
                         f'Expected velocity from {slowFile} < from {fastFile}')
        
    def test_kinematics_jerky_velocity(self):
        slowFile = 'Kinematics/slow_jerky.hdf5'
        fastFile = 'Kinematics/fast_jerky.hdf5'
        slowVelocity = get_mean_abs_velocity(slowFile, self.eval_metrics)
        fastVelocity = get_mean_abs_velocity(fastFile, self.eval_metrics)
        self.assertLess(slowVelocity, fastVelocity,
                         f'Expected velocity from {slowFile} < from {fastFile}')
    
    def test_kinematics_slow_acceleration(self):
        slowFile = 'Kinematics/slow_constant.hdf5'
        fastFile = 'Kinematics/slow_jerky.hdf5'
        slowAcceleration = get_mean_abs_acceleration(slowFile, self.eval_metrics)
        fastAcceleration = get_mean_abs_acceleration(fastFile, self.eval_metrics)
        self.assertLess(slowAcceleration, fastAcceleration,
                         f'Expected acceleration from {slowFile} < from {fastFile}')
        
    def test_kinematics_fast_acceleration(self):
        slowFile = 'Kinematics/fast_constant.hdf5'
        fastFile = 'Kinematics/fast_acc.hdf5'
        slowAcceleration = get_mean_abs_acceleration(slowFile, self.eval_metrics)
        fastAcceleration = get_mean_abs_acceleration(fastFile, self.eval_metrics)
        self.assertLess(slowAcceleration, fastAcceleration,
                         f'Expected acceleration from {slowFile} < from {fastFile}')

    def test_kinematics_jerk_slow_constant(self):
        constantFile = 'Kinematics/slow_constant.hdf5'
        slowFile = 'Kinematics/slow_jerky.hdf5'
        fastFile = 'Kinematics/fast_jerky.hdf5'
        lowJerk = get_mean_abs_jerk(constantFile, self.eval_metrics)
        slowJerky = get_mean_abs_jerk(slowFile, self.eval_metrics)
        fastJerky = get_mean_abs_jerk(fastFile, self.eval_metrics)
        self.assertLess(lowJerk, slowJerky,
                         f'Expected jerk from {constantFile} < from {slowFile}')

        self.assertLess(lowJerk, fastJerky,
                         f'Expected jerk from {constantFile} < from {fastFile}')

    def test_kinematics_jerk_fast_constant(self):
        constantFile = 'Kinematics/fast_constant.hdf5'
        slowFile = 'Kinematics/slow_jerky.hdf5'
        fastFile = 'Kinematics/fast_jerky.hdf5'
        lowJerk = get_mean_abs_jerk(constantFile, self.eval_metrics)
        slowJerky = get_mean_abs_jerk(slowFile, self.eval_metrics)
        fastJerky = get_mean_abs_jerk(fastFile, self.eval_metrics)
        self.assertLess(lowJerk, slowJerky,
                         f'Expected jerk from {lowJerk} < from {slowFile}')

        self.assertLess(lowJerk, fastJerky,
                         f'Expected jerk from {lowJerk} < from {fastFile}') #Fast jerky has a lower jerk than fast constant

    def test_no_force_removal(self):
        file = 'ForceRemove/no_force_removal.hdf5'
        force = get_max_stroke_force(file, self.eval_metrics)
        self.assertAlmostEqual(0, force, 1e-4,
                         f'Expected 0 force from {file}')
        
    def test_low_force_removal(self):
        file = 'ForceRemove/low_force_removal.hdf5'
        force = get_max_stroke_force(file, self.eval_metrics)
        self.assertGreater(force, 0,
                         f'Expected non-zero force from {file}')
        
    def test_force_removal_comparison(self):
        lowFile = 'ForceRemove/low_force_removal.hdf5'
        highFile = 'ForceRemove/high_force_removal.hdf5'
        lowForce = get_max_stroke_force(lowFile, self.eval_metrics)
        highForce = get_max_stroke_force(highFile, self.eval_metrics)
        self.assertLess(lowForce, highForce,
                         f'Expected force from {lowFile} < from {highFile}')

    def test_duration(self):
        file = 'Strokes/many_strokes.hdf5'

        _, _, v_rm = open_file(file)

        dur = ft.procedure_duration(v_rm['voxel_time_stamp'][()])
        self.eval_metrics.duration += dur
        self.assertAlmostEqual(dur, 16.45, 1,
                                f'Expected duration from {file} to be 20 sec')

    def test_curve_lengths(self):
        shortFile = 'LenCurve/short_straight.hdf5'
        longFile = 'LenCurve/long_curved.hdf5'

        shortCurve = get_mean_stroke_length(shortFile, self.eval_metrics)
        longCurve = get_mean_stroke_length(longFile, self.eval_metrics)

        self.assertLess(shortCurve, longCurve, 
                        f'Expected acceleration from {shortFile} < from {longFile}')
        
    def test_curvature(self):
        straightFile = 'LenCurve/short_straight.hdf5'
        curvyFile = 'LenCurve/long_curved.hdf5'

        straightCurve = get_mean_curvature(straightFile, self.eval_metrics)
        curvyCurve = get_mean_curvature(curvyFile, self.eval_metrics)

        self.assertLess(straightCurve, curvyCurve)

    def test_angle(self):
        acuteFile = 'Angles/45deg.hdf5'
        rightAngleFile = 'Angles/90deg.hdf5'

        acuteAngle = get_mean_angle(acuteFile)
        rightAngle = get_mean_angle(rightAngleFile)

        self.assertLess(acuteAngle, rightAngle)

    def test_no_removal(self):
        file = 'ForceRemove/no_force_removal.hdf5'
        rate = get_mean_removal_rate(file, self.eval_metrics)
        self.assertAlmostEqual(0, rate, 1e-4,
                         f'Expected 0 removal rate from {file}')

    def test_low_removal(self):
        file = 'ForceRemove/low_force_removal.hdf5'
        rate = get_mean_removal_rate(file, self.eval_metrics)
        self.assertGreater(rate, 0,
                         f'Expected non-zero rate from {file}')

    def test_removal_rate_comparison(self):
        lowFile = 'ForceRemove/low_force_removal.hdf5'
        highFile = 'ForceRemove/high_force_removal.hdf5'
        lowRate = get_mean_removal_rate(lowFile, self.eval_metrics)
        highRate = get_mean_removal_rate(highFile, self.eval_metrics)
        self.assertLess(lowRate, highRate,
                         f'Expected removal rate from {lowFile} < from {highFile}')

        

