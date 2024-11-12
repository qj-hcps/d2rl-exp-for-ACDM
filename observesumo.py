from decisionmodels.CDM.observer import Observer
import utils.globalvalues as gv
import utils.extendmath as emath
import utils.dyglobalvalues as dgv
import math


class NadeObserver(Observer):
    def __init__(self, ego_id, main_id) -> None:
        super().__init__(ego_id, main_id)

    def __del__(self):
        pass

    def get_nade_observe(self, realvehicle_id_list, prev_long=0.0, prev_lat="central"):
        """
        获取nade输入所需的OBS格式, 被测车辆
        """
        # 将被测车辆的id改成"0", 格式需要
        dgv.update_realvehicle_dict("0", dgv.pop_realvehicle(self.main_id))
        # 每辆车的观测dict
        single_obs = {
            "Ego": None,
            "Lead": None,
            "LeftLead": None,
            "RightLead": None,
            "Foll": None,
            "LeftFoll": None,
            "RightFoll": None,
        }

        # 观测自身的数据
        single_obs_ego = {
            "veh_id": None,
            "could_drive_adjacent_lane_left": False,
            "could_drive_adjacent_lane_right": False,
            "distance": 0.0,
            "heading": 90.0,  # 向前就是90
            "lane_index": 0,
            "lateral_speed": 0.0,
            "lateral_offset": 0.0,
            "prev_action": {
                "lateral": "central",
                "longitudinal": 0.0,
            },
            "position": [0.0, 0.0],
            "position3D": [0.0, 0.0, 0.0],
            "velocity": 0.0,
            "road_id": "0to1",
            "acceleration": 0.0,
        }

        # 每辆车生成一个观测值
        obs = single_obs.copy()
        obs["Ego"] = single_obs_ego.copy()
        # veh_id
        if self.ego_id == self.main_id:
            self.ego_id = "0"
        obs["Ego"]["veh_id"] = self.ego_id
        # 根据车道判断是否可变道
        vehicle = dgv.get_realvehicle(self.ego_id)
        lane_id = dgv.get_map().get_waypoint(vehicle.vehicle.get_location()).lane_id
        if lane_id == gv.LANE_ID["Left"]:
            obs["Ego"]["could_drive_adjacent_lane_right"] = True
            obs["Ego"]["lane_index"] = 1
            obs["Ego"]["position"][1] = obs["Ego"]["position3D"][1] = 46.0
        if lane_id == gv.LANE_ID["Right"]:
            obs["Ego"]["could_drive_adjacent_lane_left"] = True
            obs["Ego"]["lane_index"] = 0
            obs["Ego"]["position"][1] = obs["Ego"]["position3D"][1] = 42.0
        # 计算偏移距离
        lc_rate = vehicle.changing_lane_pace / max(len(vehicle.lane_changing_route), 1)
        obs["Ego"]["lateral_offset"] = min(
            lc_rate * gv.LANE_WIDTH, (1 - lc_rate) * gv.LANE_WIDTH
        )
        # 计算偏航角和横向绝对位置
        degree = math.atan(gv.LANE_WIDTH / vehicle.scalar_velocity) * (180 / math.pi)
        if vehicle.control_action == "SLIDE_LEFT":
            obs["Ego"]["heading"] = 90 + degree
            obs["Ego"]["lateral_speed"] = gv.LANE_WIDTH / gv.LANE_CHANGE_TIME
            obs["Ego"]["position"][1] = obs["Ego"]["position3D"][1] = (
                42.0 + lc_rate * gv.LANE_WIDTH
            )
        if vehicle.control_action == "SLIDE_RIGHT":
            obs["Ego"]["heading"] = 90 - degree
            obs["Ego"]["lateral_speed"] = gv.LANE_WIDTH / gv.LANE_CHANGE_TIME
            obs["Ego"]["position"][1] = obs["Ego"]["position3D"][1] = (
                46.0 - lc_rate * gv.LANE_WIDTH
            )

        # 上一次决策
        obs["Ego"]["prev_action"]["lateral"] = prev_lat
        obs["Ego"]["prev_action"]["longitudinal"] = prev_long

        # 计算纵向位置（假设主车固定在200）
        main_long_pos = 200
        rel_dist = emath.cal_distance_along_road(
            dgv.get_map().get_waypoint(dgv.get_realvehicle("0").vehicle.get_location()),
            dgv.get_map().get_waypoint(vehicle.vehicle.get_location()),
        )
        obs["Ego"]["position"][0] = obs["Ego"]["position3D"][0] = (
            rel_dist + main_long_pos
        )
        # 速度
        obs["Ego"]["velocity"] = vehicle.scalar_velocity
        # 加速度
        obs["Ego"]["acceleration"] = (
            gv.LON_ACC_DICT.get(vehicle.control_action)
            if type(vehicle.control_action) == str
            else vehicle.control_action
        )
        # 其他方位的车辆
        (
            min_id_front,
            _,
            min_id_back,
            _,
            min_id_sidef,
            _,
            min_id_sideb,
            _,
        ) = self.get_closest_vehicles(realvehicle_id_list)
        # 各个方位的观测
        if min_id_front:
            obs["Lead"] = self.get_other_obs(min_id_front)
        if min_id_back:
            obs["Foll"] = self.get_other_obs(min_id_back)
        if min_id_sidef and lane_id == gv.LANE_ID["Left"]:
            obs["RightLead"] = self.get_other_obs(min_id_sidef)
        if min_id_sidef and lane_id == gv.LANE_ID["Right"]:
            obs["LeftLead"] = self.get_other_obs(min_id_sidef)
        if min_id_sideb and lane_id == gv.LANE_ID["Left"]:
            obs["RightFoll"] = self.get_other_obs(min_id_sideb)
        if min_id_sideb and lane_id == gv.LANE_ID["Right"]:
            obs["LeftFoll"] = self.get_other_obs(min_id_sideb)

        return obs

    def get_other_obs(self, veh_id):
        """
        获取单个观测的单个其他车辆
        """
        vehicle = dgv.get_realvehicle(veh_id)
        # 观测其他车辆的数据
        single_obs_other = {
            "veh_id": veh_id,
            "distance": 0.0,
            "velocity": 0.0,
            "position": [0.0, 0.0],
            "heading": 90.0,
            "lane_index": 0,
            "position3D": [0.0, 0.0, 0.0],
            "acceleration": 0.0,
        }
        # 距离
        rel_long_dist = emath.cal_distance_along_road(
            dgv.get_map().get_waypoint(
                dgv.get_realvehicle(self.ego_id).vehicle.get_location()
            ),
            dgv.get_map().get_waypoint(vehicle.vehicle.get_location()),
        )
        single_obs_other["distance"] = math.sqrt(rel_long_dist**2 + gv.LANE_WIDTH**2)
        # 速度
        single_obs_other["velocity"] = vehicle.scalar_velocity
        # lane id
        lane_id = dgv.get_map().get_waypoint(vehicle.vehicle.get_location()).lane_id
        if lane_id == gv.LANE_ID["Left"]:
            single_obs_other["lane_index"] = 1
            single_obs_other["position"][1] = single_obs_other["position3D"][1] = 46.0
        if lane_id == gv.LANE_ID["Right"]:
            single_obs_other["lane_index"] = 0
            single_obs_other["position"][1] = single_obs_other["position3D"][1] = 42.0
        # 计算纵向位置（假设主车固定在200）
        main_long_pos = 200
        rel_dist = emath.cal_distance_along_road(
            dgv.get_map().get_waypoint(dgv.get_realvehicle("0").vehicle.get_location()),
            dgv.get_map().get_waypoint(vehicle.vehicle.get_location()),
        )
        single_obs_other["position"][0] = single_obs_other["position3D"][0] = (
            rel_dist + main_long_pos
        )
        # 偏航角和横向绝对位置
        lc_rate = vehicle.changing_lane_pace / max(len(vehicle.lane_changing_route), 1)
        degree = math.atan(gv.LANE_WIDTH / vehicle.scalar_velocity) * (180 / math.pi)
        if vehicle.control_action == "SLIDE_LEFT":
            single_obs_other["heading"] = 90 + degree
            single_obs_other["position"][1] = single_obs_other["position3D"][1] = (
                42.0 + lc_rate * gv.LANE_WIDTH
            )
        if vehicle.control_action == "SLIDE_RIGHT":
            single_obs_other["heading"] = 90 - degree
            single_obs_other["position"][1] = single_obs_other["position3D"][1] = (
                46.0 - lc_rate * gv.LANE_WIDTH
            )
        # 加速度
        single_obs_other["acceleration"] = (
            gv.LON_ACC_DICT.get(vehicle.control_action)
            if type(vehicle.control_action) == str
            else vehicle.control_action
        )
        return single_obs_other

    def get_closest_vehicles(self, realvehicle_id_list, mode="real"):
        """
        筛选出主车前后侧方最近车辆
        mode = real || virtual, 代表传入的字典是realvehicle还是virtualvehicle
        """
        self.close_vehicle_id_list = self.get_close_vehicle_id_list(realvehicle_id_list)
        ego_vehicle = dgv.get_realvehicle(self.ego_id)
        min_dist_front, min_dist_back, min_dist_sidef, min_dist_sideb = (
            1e9,
            -1e9,
            1e9,
            -1e9,
        )
        min_id_front, min_id_back, min_id_sidef, min_id_sideb = None, None, None, None
        if mode == "real":
            ego_wp = dgv.get_map().get_waypoint(ego_vehicle.vehicle.get_location())
        elif mode == "virtual":
            ego_wp = ego_vehicle.waypoint
        else:
            raise ValueError("The mode value is wrong!")
        # 对每辆车判断是否处于同一车道且前后距离最短
        for rvid in self.close_vehicle_id_list:
            if rvid != self.ego_id:
                vehicle = dgv.get_realvehicle(rvid)
                if mode == "real":
                    veh_wp = dgv.get_map().get_waypoint(vehicle.vehicle.get_location())
                elif mode == "virtual":
                    veh_wp = vehicle.waypoint
                else:
                    raise ValueError("The mode value is wrong!")
                rel_distance = emath.cal_distance_along_road(ego_wp, veh_wp)
                if veh_wp.lane_id == ego_wp.lane_id:
                    # 前方车辆
                    if 0 < rel_distance < min_dist_front:
                        min_dist_front = rel_distance
                        min_id_front = rvid
                    # 后方车辆
                    if min_dist_back < rel_distance < 0:
                        min_dist_back = rel_distance
                        min_id_back = rvid
                if veh_wp.lane_id != ego_wp.lane_id:
                    # 侧前方车辆
                    if 0 < rel_distance < min_dist_front:
                        min_dist_sidef = rel_distance
                        min_id_sidef = rvid
                    # 侧后方车辆
                    if min_dist_sideb < rel_distance < 0:
                        min_dist_sideb = rel_distance
                        min_id_sideb = rvid
        # 返回前方最近车辆与其相对距离
        return (
            min_id_front,
            min_dist_front,
            min_id_back,
            min_dist_back,
            min_id_sidef,
            min_dist_sidef,
            min_id_sideb,
            min_dist_sideb,
        )
