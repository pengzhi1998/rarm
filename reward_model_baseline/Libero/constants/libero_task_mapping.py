#!/usr/bin/env python3
"""
LIBERO Task Mapping

Maps task_id (0-9) to task names for a given suite.
Switch suites by changing SUITE below.

Supported suites:
  "libero_10"      — diverse tasks across kitchen, living room, study
  "libero_spatial" — 10 pick-and-place variations in a single scene
  "libero_goal"    — 10 goal-conditioned tasks
  "libero_object"  — 10 pick-and-place tasks with different objects
  "libero_90"      — 90 tasks across kitchen, living room, study

Task names sourced from: lib/LIBERO/libero/libero/benchmark/libero_suite_task_map.py
"""

import re


def _to_language(env_name: str) -> str:
    """Strip scene prefix (e.g. KITCHEN_SCENE3_, LIVING_ROOM_SCENE2_) and humanise."""
    stripped = re.sub(r'^.+_SCENE\d+_', '', env_name)
    return stripped.replace("_", " ")

SUITE = "libero_10"  # Change this to switch suites: "libero_10", "libero_spatial", "libero_goal", "libero_object", "libero_90"

# ── libero_10 ─────────────────────────────────────────────────────────────────
LIBERO_10_TASK_ID_TO_ENV_NAME = {
    0: "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket",
    1: "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket",
    2: "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
    3: "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    4: "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate",
    5: "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    6: "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate",
    7: "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket",
    8: "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove",
    9: "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
}

# ── libero_spatial ─────────────────────────────────────────────────────────────
LIBERO_SPATIAL_TASK_ID_TO_ENV_NAME = {
    0: "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    1: "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    2: "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
    3: "pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate",
    4: "pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate",
    5: "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate",
    6: "pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate",
    7: "pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate",
    8: "pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate",
    9: "pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate",
}

# ── libero_goal ───────────────────────────────────────────────────────────────
# Order must match lib/LIBERO/libero/libero/benchmark/libero_suite_task_map.py
LIBERO_GOAL_TASK_ID_TO_ENV_NAME = {
    0: "open_the_middle_drawer_of_the_cabinet",
    1: "put_the_bowl_on_the_stove",
    2: "put_the_wine_bottle_on_top_of_the_cabinet",
    3: "open_the_top_drawer_and_put_the_bowl_inside",
    4: "put_the_bowl_on_top_of_the_cabinet",
    5: "push_the_plate_to_the_front_of_the_stove",
    6: "put_the_cream_cheese_in_the_bowl",
    7: "turn_on_the_stove",
    8: "put_the_bowl_on_the_plate",
    9: "put_the_wine_bottle_on_the_rack",
}

# ── libero_object ─────────────────────────────────────────────────────────────
# Order must match lib/LIBERO/libero/libero/benchmark/libero_suite_task_map.py
LIBERO_OBJECT_TASK_ID_TO_ENV_NAME = {
    0: "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
    1: "pick_up_the_cream_cheese_and_place_it_in_the_basket",
    2: "pick_up_the_salad_dressing_and_place_it_in_the_basket",
    3: "pick_up_the_bbq_sauce_and_place_it_in_the_basket",
    4: "pick_up_the_ketchup_and_place_it_in_the_basket",
    5: "pick_up_the_tomato_sauce_and_place_it_in_the_basket",
    6: "pick_up_the_butter_and_place_it_in_the_basket",
    7: "pick_up_the_milk_and_place_it_in_the_basket",
    8: "pick_up_the_chocolate_pudding_and_place_it_in_the_basket",
    9: "pick_up_the_orange_juice_and_place_it_in_the_basket",
}

# ── libero_90 ─────────────────────────────────────────────────────────────────
# Order must match lib/LIBERO/libero/libero/benchmark/libero_suite_task_map.py
LIBERO_90_TASK_ID_TO_ENV_NAME = {
    0:  "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet",
    1:  "KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_and_put_the_black_bowl_on_top_of_it",
    2:  "KITCHEN_SCENE10_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
    3:  "KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it",
    4:  "KITCHEN_SCENE10_put_the_butter_at_the_front_in_the_top_drawer_of_the_cabinet_and_close_it",
    5:  "KITCHEN_SCENE10_put_the_chocolate_pudding_in_the_top_drawer_of_the_cabinet_and_close_it",
    6:  "KITCHEN_SCENE1_open_the_bottom_drawer_of_the_cabinet",
    7:  "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet",
    8:  "KITCHEN_SCENE1_open_the_top_drawer_of_the_cabinet_and_put_the_bowl_in_it",
    9:  "KITCHEN_SCENE1_put_the_black_bowl_on_the_plate",
    10: "KITCHEN_SCENE1_put_the_black_bowl_on_top_of_the_cabinet",
    11: "KITCHEN_SCENE2_open_the_top_drawer_of_the_cabinet",
    12: "KITCHEN_SCENE2_put_the_black_bowl_at_the_back_on_the_plate",
    13: "KITCHEN_SCENE2_put_the_black_bowl_at_the_front_on_the_plate",
    14: "KITCHEN_SCENE2_put_the_middle_black_bowl_on_the_plate",
    15: "KITCHEN_SCENE2_put_the_middle_black_bowl_on_top_of_the_cabinet",
    16: "KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle",
    17: "KITCHEN_SCENE2_stack_the_middle_black_bowl_on_the_back_black_bowl",
    18: "KITCHEN_SCENE3_put_the_frying_pan_on_the_stove",
    19: "KITCHEN_SCENE3_put_the_moka_pot_on_the_stove",
    20: "KITCHEN_SCENE3_turn_on_the_stove",
    21: "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_frying_pan_on_it",
    22: "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet",
    23: "KITCHEN_SCENE4_close_the_bottom_drawer_of_the_cabinet_and_open_the_top_drawer",
    24: "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet",
    25: "KITCHEN_SCENE4_put_the_black_bowl_on_top_of_the_cabinet",
    26: "KITCHEN_SCENE4_put_the_wine_bottle_in_the_bottom_drawer_of_the_cabinet",
    27: "KITCHEN_SCENE4_put_the_wine_bottle_on_the_wine_rack",
    28: "KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet",
    29: "KITCHEN_SCENE5_put_the_black_bowl_in_the_top_drawer_of_the_cabinet",
    30: "KITCHEN_SCENE5_put_the_black_bowl_on_the_plate",
    31: "KITCHEN_SCENE5_put_the_black_bowl_on_top_of_the_cabinet",
    32: "KITCHEN_SCENE5_put_the_ketchup_in_the_top_drawer_of_the_cabinet",
    33: "KITCHEN_SCENE6_close_the_microwave",
    34: "KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug",
    35: "KITCHEN_SCENE7_open_the_microwave",
    36: "KITCHEN_SCENE7_put_the_white_bowl_on_the_plate",
    37: "KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate",
    38: "KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove",
    39: "KITCHEN_SCENE8_turn_off_the_stove",
    40: "KITCHEN_SCENE9_put_the_frying_pan_on_the_cabinet_shelf",
    41: "KITCHEN_SCENE9_put_the_frying_pan_on_top_of_the_cabinet",
    42: "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf",
    43: "KITCHEN_SCENE9_put_the_white_bowl_on_top_of_the_cabinet",
    44: "KITCHEN_SCENE9_turn_on_the_stove",
    45: "KITCHEN_SCENE9_turn_on_the_stove_and_put_the_frying_pan_on_it",
    46: "LIVING_ROOM_SCENE1_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
    47: "LIVING_ROOM_SCENE1_pick_up_the_cream_cheese_box_and_put_it_in_the_basket",
    48: "LIVING_ROOM_SCENE1_pick_up_the_ketchup_and_put_it_in_the_basket",
    49: "LIVING_ROOM_SCENE1_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
    50: "LIVING_ROOM_SCENE2_pick_up_the_alphabet_soup_and_put_it_in_the_basket",
    51: "LIVING_ROOM_SCENE2_pick_up_the_butter_and_put_it_in_the_basket",
    52: "LIVING_ROOM_SCENE2_pick_up_the_milk_and_put_it_in_the_basket",
    53: "LIVING_ROOM_SCENE2_pick_up_the_orange_juice_and_put_it_in_the_basket",
    54: "LIVING_ROOM_SCENE2_pick_up_the_tomato_sauce_and_put_it_in_the_basket",
    55: "LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray",
    56: "LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray",
    57: "LIVING_ROOM_SCENE3_pick_up_the_cream_cheese_and_put_it_in_the_tray",
    58: "LIVING_ROOM_SCENE3_pick_up_the_ketchup_and_put_it_in_the_tray",
    59: "LIVING_ROOM_SCENE3_pick_up_the_tomato_sauce_and_put_it_in_the_tray",
    60: "LIVING_ROOM_SCENE4_pick_up_the_black_bowl_on_the_left_and_put_it_in_the_tray",
    61: "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray",
    62: "LIVING_ROOM_SCENE4_pick_up_the_salad_dressing_and_put_it_in_the_tray",
    63: "LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray",
    64: "LIVING_ROOM_SCENE4_stack_the_right_bowl_on_the_left_bowl_and_place_them_in_the_tray",
    65: "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_left_plate",
    66: "LIVING_ROOM_SCENE5_put_the_red_mug_on_the_right_plate",
    67: "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate",
    68: "LIVING_ROOM_SCENE5_put_the_yellow_and_white_mug_on_the_right_plate",
    69: "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate",
    70: "LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_right_of_the_plate",
    71: "LIVING_ROOM_SCENE6_put_the_red_mug_on_the_plate",
    72: "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate",
    73: "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
    74: "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
    75: "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
    76: "STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy",
    77: "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy",
    78: "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
    79: "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
    80: "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
    81: "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy",
    82: "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy",
    83: "STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_right_compartment_of_the_caddy",
    84: "STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy",
    85: "STUDY_SCENE3_pick_up_the_white_mug_and_place_it_to_the_right_of_the_caddy",
    86: "STUDY_SCENE4_pick_up_the_book_in_the_middle_and_place_it_on_the_cabinet_shelf",
    87: "STUDY_SCENE4_pick_up_the_book_on_the_left_and_place_it_on_top_of_the_shelf",
    88: "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_on_the_cabinet_shelf",
    89: "STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf",
}

# ── Natural language descriptions (scene prefix stripped) ─────────────────────
LIBERO_10_TASK_ID_TO_LANGUAGE = {k: _to_language(v) for k, v in LIBERO_10_TASK_ID_TO_ENV_NAME.items()}
LIBERO_SPATIAL_TASK_ID_TO_LANGUAGE = {k: _to_language(v) for k, v in LIBERO_SPATIAL_TASK_ID_TO_ENV_NAME.items()}
LIBERO_GOAL_TASK_ID_TO_LANGUAGE = {k: _to_language(v) for k, v in LIBERO_GOAL_TASK_ID_TO_ENV_NAME.items()}
LIBERO_OBJECT_TASK_ID_TO_LANGUAGE = {k: _to_language(v) for k, v in LIBERO_OBJECT_TASK_ID_TO_ENV_NAME.items()}
LIBERO_90_TASK_ID_TO_LANGUAGE = {k: _to_language(v) for k, v in LIBERO_90_TASK_ID_TO_ENV_NAME.items()}

# ── Active suite ───────────────────────────────────────────────────────────────
_SUITE_MAP = {
    "libero_10":      LIBERO_10_TASK_ID_TO_ENV_NAME,
    "libero_spatial": LIBERO_SPATIAL_TASK_ID_TO_ENV_NAME,
    "libero_goal":    LIBERO_GOAL_TASK_ID_TO_ENV_NAME,
    "libero_object":  LIBERO_OBJECT_TASK_ID_TO_ENV_NAME,
    "libero_90":      LIBERO_90_TASK_ID_TO_ENV_NAME,
}

_LANGUAGE_MAP = {
    "libero_10":      LIBERO_10_TASK_ID_TO_LANGUAGE,
    "libero_spatial": LIBERO_SPATIAL_TASK_ID_TO_LANGUAGE,
    "libero_goal":    LIBERO_GOAL_TASK_ID_TO_LANGUAGE,
    "libero_object":  LIBERO_OBJECT_TASK_ID_TO_LANGUAGE,
    "libero_90":      LIBERO_90_TASK_ID_TO_LANGUAGE,
}

if SUITE not in _SUITE_MAP:
    raise ValueError(f"Unknown SUITE '{SUITE}'. Choose from: {list(_SUITE_MAP.keys())}")

TASK_ID_TO_ENV_NAME = _SUITE_MAP[SUITE]
TASK_ID_TO_LANGUAGE = _LANGUAGE_MAP[SUITE]

# Reverse mapping: env_name → task_id
ENV_NAME_TO_TASK_ID = {v: k for k, v in TASK_ID_TO_ENV_NAME.items()}

NUM_TASKS = len(TASK_ID_TO_ENV_NAME)
