import sys
from decisionmodels.D2RL.envs.nade import NADE
from decisionmodels.D2RL.controller.treesearchnadecontroller import (
    TreeSearchNADEBackgroundController,
)
from decisionmodels.D2RL.controller.nadeglobalcontroller import NADEBVGlobalController
from decisionmodels.D2RL.conf import conf
from decisionmodels.D2RL.observesumo import NadeObserver
import random
import utils.globalvalues as gv


conf.experiment_config["mode"] = "D2RL"  # args.mode
conf.experiment_config["experiment_name"] = (
    "2lane_400m_D2RL_testing"  # args.experiment_name
)
if conf.experiment_config["mode"] == "NDE":
    conf.simulation_config["epsilon_setting"] = "fixed"  # not use d2rl-based agent
elif conf.experiment_config["mode"] == "D2RL":
    conf.simulation_config["epsilon_setting"] = "drl"  # use d2rl-based agent
elif conf.experiment_config["mode"] == "behavior_policy":
    conf.simulation_config["epsilon_setting"] = "fixed"  # use behavior policy

print(
    f"Using mode {conf.experiment_config['mode']}, epsilon_setting {conf.simulation_config['epsilon_setting']}"
)
# If running D2RL experiments, then load the D2RL agent
d2rl_agent_path = "./decisionmodels/D2RL/checkpoints/2lane_400m/model.pt"
if conf.simulation_config["epsilon_setting"] == "drl":
    try:
        conf.discriminator_agent = conf.load_discriminator_agent(
            checkpoint_path=d2rl_agent_path
        )
    except:
        print("Time out, shutting down")
        sys.exit(0)


def action_idx_to_action(idx):
    """
    33个action
    """
    low = -4
    high = 2
    gate = 0.2
    if idx == 0:
        return "SLIDE_LEFT"
    if idx == 1:
        return "SLIDE_RIGHT"
    return low + (idx - 2) * gate


class D2RLDecisionModel:
    def __init__(self, map, ego_id, main_id) -> None:
        self._map = map
        self._ego_id = ego_id
        self._main_id = main_id

    def single_decision(self, bv_pdf_dict, bv_action_idx_dict):
        if self._ego_id == self._main_id:
            self._ego_id == "0"
        bv_pdf = bv_pdf_dict.get(self._ego_id)
        if bv_pdf is None:
            return "MAINTAIN"
        if bv_action_idx_dict.get(self._ego_id) != None:
            return action_idx_to_action(bv_action_idx_dict.get(self._ego_id))
        else:
            random.seed(2024)
            return action_idx_to_action(
                random.choices(list(range(len(bv_pdf))), weights=bv_pdf)[0]
            )


class D2RLGlobalDecisionModel:
    def __init__(self, map, main_id) -> None:
        self._env = NADE(
            BVController=TreeSearchNADEBackgroundController,
            cav_model=conf.experiment_config["AV_model"],
        )
        self._controller = NADEBVGlobalController(self._env)
        self._map = map
        self._main_id = main_id

    def global_decision(self, full_vehicle_id_dict, close_vehicle_id_list, time_step):
        """
        全局控制决策
        """
        obs_dict = {}
        for veh_id in close_vehicle_id_list:
            vehicle = full_vehicle_id_dict.get(veh_id)
            #
            if vehicle._nade_observer == None:
                vehicle._nade_observer = NadeObserver(veh_id, self._main_id, self._map)
            # 前一次决策
            prev_lat = "central"
            if vehicle._control_action == "SLIDE_LEFT":
                prev_lat = "left"
            if vehicle._control_action == "SLIDE_RIGHT":
                prev_lat = "right"
            prev_lon = (
                gv.LON_ACC_DICT.get(vehicle._control_action)
                if type(vehicle._control_action) == str
                else vehicle._control_action
            )
            if veh_id == self._main_id:
                obs_dict["0"] = vehicle._nade_observer.get_nade_observe(
                    full_vehicle_id_dict, prev_lon, prev_lat
                )
            else:
                obs_dict[veh_id] = vehicle._nade_observer.get_nade_observe(
                    full_vehicle_id_dict, prev_lon, prev_lat
                )
        (
            bv_pdf_dict,
            bv_action_idx_list,
            weight_list,
            max_vehicle_criticality,
            ndd_possi_list,
            IS_possi_list,
            controlled_bvs_id_list,
            controlled_bvs_list,
            vehicle_criticality_list,
            discriminator_input,
        ) = self._controller.select_controlled_bv_and_action_exp(
            "0", obs_dict, time_step
        )
        bv_action_idx_dict = dict(list(zip(controlled_bvs_id_list, bv_action_idx_list)))
        for key in bv_action_idx_dict:
            if bv_action_idx_dict[key] != None:
                print(bv_action_idx_dict)
                break
        return bv_pdf_dict, bv_action_idx_dict
