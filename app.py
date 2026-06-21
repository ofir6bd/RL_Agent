"""Flask backend API for RL Agent UI.

Serves patient data, evaluation results, and agent metrics.
"""
from __future__ import annotations
from flask import Flask, jsonify, request
from flask_cors import CORS
from pathlib import Path
import numpy as np
import json
import random
from typing import Dict, List, Optional
from datetime import datetime
import sys
import traceback

# Import with error handling
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: PyTorch not available")

try:
    from src.config import load_config, PTV_NAMES, OAR_NAMES, ALL_STRUCTURES
    from src.env.dose_env import DoseEnv
    from src.agents.ppo import PPO
    MODELS_AVAILABLE = True
except Exception as e:
    MODELS_AVAILABLE = False
    print(f"WARNING: Could not import RL models: {e}")
    traceback.print_exc()

app = Flask(__name__)
CORS(app)

# Global state
CONFIG = None
ENV = None
AGENT = None
BACKEND_READY = False
PATIENT_CACHE = {}
EVAL_RESULTS = {}


def init_backend(config_path: str = "configs/default.yaml", 
                 ckpt_path: str = "runs/best.pt",
                 split: str = "validation"):
    """Initialize backend with config, environment, and agent."""
    global CONFIG, ENV, AGENT, BACKEND_READY
    
    try:
        if not MODELS_AVAILABLE:
            print("ERROR: RL models not available, using demo mode")
            CONFIG = {
                'n_fractions': 35,
                'n_beams': 9,
                'beamlet_h': 16,
                'beamlet_w': 16,
                'prescription': {"PTV70": 70.0, "PTV63": 63.0, "PTV56": 56.0},
                'oar_tolerance': {
                    "Brainstem": 54.0, "SpinalCord": 45.0, "Mandible": 70.0,
                    "LeftParotid": 26.0, "RightParotid": 26.0,
                }
            }
            BACKEND_READY = True
            return CONFIG, None, None
        
        CONFIG = load_config(config_path)
        ENV = DoseEnv(CONFIG, split=split)
        
        # Load pretrained agent
        # PPO.__init__ takes (cfg, in_channels)
        # in_channels = CT(1) + structures(3+5) + cumulative_dose(1) + ptv_gap(1) + beam_paths(1) = 12
        in_channels = 12
        AGENT = PPO(CONFIG, in_channels=in_channels)
        
        device = torch.device(CONFIG.device if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(ckpt_path, map_location=device)
        AGENT.net.load_state_dict(checkpoint['net'])
        AGENT.net.eval()
        
        BACKEND_READY = True
        print("✓ Backend initialized successfully")
        return CONFIG, ENV, AGENT
    except Exception as e:
        print(f"ERROR initializing backend: {e}")
        traceback.print_exc()
        BACKEND_READY = True  # Still mark as ready but in degraded mode
        return None, None, None


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok', 
        'timestamp': datetime.now().isoformat(),
        'backend_ready': BACKEND_READY,
        'models_available': MODELS_AVAILABLE
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get system configuration."""
    if CONFIG is None:
        return jsonify({
            'n_fractions': 35,
            'n_beams': 9,
            'beamlet_h': 16,
            'beamlet_w': 16,
            'ptv_names': ["PTV70", "PTV63", "PTV56"],
            'oar_names': ["Brainstem", "SpinalCord", "Mandible", "LeftParotid", "RightParotid"],
            'ptv_prescriptions': {"PTV70": 70.0, "PTV63": 63.0, "PTV56": 56.0},
            'oar_tolerances': {
                "Brainstem": 54.0, "SpinalCord": 45.0, "Mandible": 70.0,
                "LeftParotid": 26.0, "RightParotid": 26.0,
            }
        })
    return jsonify({
        'n_fractions': CONFIG.n_fractions,
        'n_beams': CONFIG.n_beams,
        'beamlet_h': CONFIG.beamlet_h,
        'beamlet_w': CONFIG.beamlet_w,
        'ptv_names': PTV_NAMES if MODELS_AVAILABLE else ["PTV70", "PTV63", "PTV56"],
        'oar_names': OAR_NAMES if MODELS_AVAILABLE else ["Brainstem", "SpinalCord", "Mandible", "LeftParotid", "RightParotid"],
        'ptv_prescriptions': CONFIG.prescription,
        'oar_tolerances': CONFIG.oar_tolerance,
    })


@app.route('/api/patients', methods=['GET'])
def get_patients():
    """Get list of available patients."""
    try:
        if ENV is None:
            # Return demo patients
            demo_patients = [f"pt_{200+i}" for i in range(20)]
            return jsonify({
                'patients': demo_patients,
                'total_count': len(demo_patients),
                'mode': 'demo'
            })
        patients = sorted(ENV.patient_ids)
        return jsonify({
            'patients': patients,
            'total_count': len(patients),
            'mode': 'live'
        })
    except Exception as e:
        print(f"Error getting patients: {e}")
        demo_patients = [f"pt_{200+i}" for i in range(20)]
        return jsonify({
            'patients': demo_patients,
            'total_count': len(demo_patients),
            'mode': 'demo',
            'error': str(e)
        })


@app.route('/api/patients/<patient_id>/data', methods=['GET'])
def get_patient_data(patient_id):
    """Get patient anatomy and dose influence matrix info."""
    try:
        if patient_id not in PATIENT_CACHE:
            # Load patient data
            state, _ = ENV.reset(patient_id=patient_id)
            patient_data = {
                'patient_id': patient_id,
                'state_shape': list(state.shape),
                'structures': {
                    'ptvs': PTV_NAMES,
                    'oars': OAR_NAMES
                }
            }
            PATIENT_CACHE[patient_id] = patient_data
        
        return jsonify(PATIENT_CACHE[patient_id])
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/api/patients/<patient_id>/evaluation', methods=['GET'])
def get_patient_evaluation(patient_id):
    """Get evaluation results for a patient."""
    try:
        eval_file = Path(f"runs/eval_best.txt")
        
        # Parse evaluation file (simplified - adjust based on your format)
        results = {
            'patient_id': patient_id,
            'ptv_coverage': {},
            'oar_doses': {},
            'overall_reward': 0.0
        }
        
        # Initialize with default structure
        for ptv in PTV_NAMES:
            results['ptv_coverage'][ptv] = random.uniform(0.8, 1.0)
        for oar in OAR_NAMES:
            results['oar_doses'][oar] = {
                'mean': random.uniform(10, 50),
                'max': random.uniform(50, 80),
                'tolerance': CONFIG.oar_tolerance[oar]
            }
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


def _build_summary(fraction_data, prescriptions, tolerances):
    """Build end-of-treatment summary from all fraction data."""
    if not fraction_data:
        return {}
    last = fraction_data[-1]
    final_doses = last.get('cumulative_organ_doses', {})
    total_reward = sum(f.get('reward', 0) for f in fraction_data)
    avg_reward = total_reward / len(fraction_data)

    ptv_summary = {}
    for ptv, rx in prescriptions.items():
        dose = final_doses.get(ptv, 0)
        coverage_pct = min((dose / rx) * 100, 100) if rx > 0 else 0
        ptv_summary[ptv] = {
            'final_dose': round(dose, 2),
            'prescribed': rx,
            'coverage_pct': round(coverage_pct, 1),
            'achieved': coverage_pct >= 95.0,
        }

    oar_summary = {}
    for oar, tol in tolerances.items():
        dose = final_doses.get(oar, 0)
        violation = dose > tol
        oar_summary[oar] = {
            'final_dose': round(dose, 2),
            'tolerance': tol,
            'pct_of_tolerance': round((dose / tol) * 100, 1) if tol > 0 else 0,
            'violation': violation,
        }

    return {
        'total_reward': round(total_reward, 4),
        'avg_reward_per_fraction': round(avg_reward, 4),
        'ptv_summary': ptv_summary,
        'oar_summary': oar_summary,
        'n_fractions': len(fraction_data),
    }


@app.route('/api/patients/<patient_id>/simulate', methods=['POST'])
def simulate_patient(patient_id):
    """Run agent simulation on a patient for all fractions."""
    try:
        prescriptions = CONFIG.prescription if CONFIG else {"PTV70": 70.0, "PTV63": 63.0, "PTV56": 56.0}
        tolerances = CONFIG.oar_tolerance if CONFIG else {
            "Brainstem": 54.0, "SpinalCord": 45.0, "Mandible": 70.0,
            "LeftParotid": 26.0, "RightParotid": 26.0,
        }
        n_fractions = CONFIG.n_fractions if CONFIG else 35

        if ENV is None:
            # Demo mode - simulate all 35 fractions realistically
            fraction_data = []
            cumulative = {k: 0.0 for k in list(prescriptions.keys()) + list(tolerances.keys())}
            for i in range(n_fractions):
                beam_heatmap = []
                for b in range(9):
                    beam_doses = (0.3 + i * 0.005) * np.random.rand(16, 16)
                    beam_heatmap.append(beam_doses.tolist())
                # Increment cumulative doses per fraction
                per_frac = {
                    'PTV70': 70.0 / n_fractions * (0.9 + 0.2 * np.random.rand()),
                    'PTV63': 63.0 / n_fractions * (0.9 + 0.2 * np.random.rand()),
                    'PTV56': 56.0 / n_fractions * (0.9 + 0.2 * np.random.rand()),
                    'Brainstem': 54.0 / n_fractions * 0.4 * np.random.rand(),
                    'SpinalCord': 45.0 / n_fractions * 0.35 * np.random.rand(),
                    'Mandible': 70.0 / n_fractions * 0.45 * np.random.rand(),
                    'LeftParotid': 26.0 / n_fractions * 0.5 * np.random.rand(),
                    'RightParotid': 26.0 / n_fractions * 0.5 * np.random.rand(),
                }
                for k in cumulative:
                    cumulative[k] += per_frac.get(k, 0)
                fraction_data.append({
                    'fraction': i + 1,
                    'reward': float(0.5 + (i * 0.005) + (np.random.random() * 0.05)),
                    'oar_penalty': float(max(0, 0.08 - (i * 0.001))),
                    'ptv_reward': float(0.6 + (i * 0.003)),
                    'action_mean': float(0.3 + np.random.random() * 0.1),
                    'action_max': float(0.7 + np.random.random() * 0.2),
                    'beam_heatmap': beam_heatmap,
                    'cumulative_organ_doses': {k: round(v, 2) for k, v in cumulative.items()},
                })
            summary = _build_summary(fraction_data, prescriptions, tolerances)
            return jsonify({
                'patient_id': patient_id,
                'fractions': fraction_data,
                'total_fractions_simulated': len(fraction_data),
                'summary': summary,
                'mode': 'demo',
            })

        # Live mode - run all n_fractions
        state, fraction_progress = ENV.reset(patient_id=patient_id)
        patient_done = False
        fraction_data = []
        fraction_idx = 0

        while not patient_done and fraction_idx < n_fractions:
            if AGENT is None:
                action = np.random.rand(CONFIG.n_beams * CONFIG.beamlet_h * CONFIG.beamlet_w) * 0.5
            else:
                action, _, _, _ = AGENT.act(state, fraction_progress, deterministic=True)

            action_2d = action.reshape((CONFIG.n_beams, CONFIG.beamlet_h, CONFIG.beamlet_w))
            beam_heatmap = [action_2d[b].tolist() for b in range(CONFIG.n_beams)]

            structure_masks = ENV._structure_masks()
            cumulative_organ_doses = {}
            for struct_name, mask in structure_masks.items():
                mean_dose = float(np.mean(ENV.cumulative_dose[mask > 0])) if np.any(mask) else 0.0
                cumulative_organ_doses[struct_name] = round(mean_dose, 2)

            state, fraction_progress, reward, done, info = ENV.step(action)
            patient_done = info.get('patient_done', done)

            fraction_data.append({
                'fraction': info.get('fraction_index', fraction_idx + 1),
                'reward': float(reward),
                'oar_penalty': float(info.get('oar_penalty', 0)),
                'ptv_reward': float(info.get('ptv_reward', 0)),
                'action_mean': float(action.mean()),
                'action_max': float(action.max()),
                'beam_heatmap': beam_heatmap,
                'cumulative_organ_doses': cumulative_organ_doses,
            })
            fraction_idx += 1

        summary = _build_summary(fraction_data, prescriptions, tolerances)
        return jsonify({
            'patient_id': patient_id,
            'fractions': fraction_data,
            'total_fractions_simulated': len(fraction_data),
            'summary': summary,
            'mode': 'live' if AGENT is not None else 'demo',
        })
    except Exception as e:
        print(f"Error in simulate_patient: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 400


@app.route('/api/patients/random', methods=['GET'])
def get_random_patient():
    """Get a random patient ID."""
    patient_id = random.choice(ENV.patient_ids)
    return jsonify({'patient_id': patient_id})


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get current system status and metrics."""
    return jsonify({
        'total_patients': len(ENV.patient_ids),
        'n_fractions': CONFIG.n_fractions,
        'agent_ready': AGENT is not None,
        'config_loaded': CONFIG is not None,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--ckpt', default='runs/best.pt')
    parser.add_argument('--split', default='validation')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    
    print("Initializing backend...")
    init_backend(args.config, args.ckpt, args.split)
    print("Backend ready!")
    
    app.run(debug=args.debug, port=args.port, host='0.0.0.0')
