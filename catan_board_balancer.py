# -*- coding: utf-8 -*-
import copy
import math
import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import RegularPolygon

plt.rcParams["figure.dpi"] = 300
fig, ax = plt.subplots(1)
ax.set_aspect("equal")
ax.axis([-7, 7, -7, 7])
plt.axis("off")


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    if dx == dy == 0:
        return np.hypot(px - x1, py - y1)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.hypot(px - proj_x, py - proj_y)


def get_formation(version='original'):
    hex_rad = 2 / np.sqrt(3)
    if version == 'original':
        # Hexagon locations using a doubled coordinate system
        # x location, y location, ID, type of tile, dice roll number
        double_coord = [
            [-2, 2, 0, "", 0],
            [0, 2, 1, "", 0],
            [2, 2, 2, "", 0],
            [-3, 1, 3, "", 0],
            [-1, 1, 4, "", 0],
            [1, 1, 5, "", 0],
            [3, 1, 6, "", 0],
            [-4, 0, 7, "", 0],
            [-2, 0, 8, "", 0],
            [0, 0, 9, "", 0],
            [2, 0, 10, "", 0],
            [4, 0, 11, "", 0],
            [-3, -1, 12, "", 0],
            [-1, -1, 13, "", 0],
            [1, -1, 14, "", 0],
            [3, -1, 15, "", 0],
            [-2, -2, 16, "", 0],
            [0, -2, 17, "", 0],
            [2, -2, 18, "", 0],
        ]

        # port locations
        # x location (pier 1), y location (pier 1), x location (pier 2), y location (pier 2) ID, type of port
        port_coord = [
            [-1, 3.5 * hex_rad, 0, 4 * hex_rad, 0, ""],
            [2, 4 * hex_rad, 3, 3.5 * hex_rad, 1, ""],
            [4, 2 * hex_rad, 4, hex_rad, 2, ""],
            [4, -hex_rad, 4, -2 * hex_rad, 3, ""],
            [3, -3.5 * hex_rad, 2, -4 * hex_rad, 4, ""],
            [0, -4 * hex_rad, -1, -3.5 * hex_rad, 5, ""],
            [-3, -2.5 * hex_rad, -4, -2 * hex_rad, 6, ""],
            [-5, -0.5 * hex_rad, -5, +0.5 * hex_rad, 7, ""],
            [-4, 2 * hex_rad, -3, 2.5 * hex_rad, 8, ""],
        ]

        # List of tiles that are too close to the port to match the port's resource type
        # Note tile ID 20 is used as filler/placeholder where less than 5 banned tiles are needed
        # Port ID, Tile ID 1,  Tile ID 2,  Tile ID 3,  Tile ID 4,  Tile ID 5
        port_banned_tiles = [
            [0, 0, 1, 2, 4, 5],
            [1, 1, 2, 5, 6, 20],
            [2, 2, 5, 6, 10, 11],
            [3, 10, 11, 14, 15, 18],
            [4, 14, 15, 17, 18, 20],
            [5, 13, 14, 16, 17, 18],
            [6, 7, 8, 12, 13, 16],
            [7, 3, 7, 8, 12, 20],
            [8, 0, 3, 4, 7, 8],
        ]

        list_of_ports_start = ["wood", "?", "wheat", "stone", "?", "sheep", "?", "?", "brick"]
        list_of_ports = copy.deepcopy(list_of_ports_start)
        default_port_locations = 1  #

        list_of_roll_numbers_start = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
        list_of_tiles_start = [
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "wheat",
            "wheat",
            "wheat",
            "wheat",
            "wood",
            "wood",
            "wood",
            "wood",
            "stone",
            "stone",
            "stone",
            "brick",
            "brick",
            "brick",
        ]
    elif version == 'got':
        double_coord = [
            [-3, 0, 0, "", 0],
            [-2, 1, 1, "", 0],
            [-1, 2, 2, "", 0],
            [0, 3, 3, "", 0],
            [2, 3, 4, "", 0],
            [1, 2, 5, "", 0],
            [0, 1, 6, "", 0],
            [-1, 0, 7, "", 0],
            [1, 0, 8, "", 0],
            [2, 1, 9, "", 0],
            [3, 2, 10, "", 0],
            [4, 3, 11, "", 0],
            [6, 3, 12, "", 0],
            [5, 2, 13, "", 0],
            [4, 1, 14, "", 0],
            [3, 0, 15, "", 0],
            [5, 0, 16, "", 0],
            [6, 1, 17, "", 0],
            [7, 2, 18, "", 0],
            [8, 1, 19, "", 0],
            [7, 0, 20, "", 0],
        ]

        # Step 1: Find min y value (lowest row - is upside down here)
        bottom_y = min(y for x, y, *_ in double_coord)

        # Step 2: Collect tiles on the bottom row
        bottom_tiles = sorted([(x, y) for x, y, *_ in double_coord if y == bottom_y])

        # Step 3: Place ports along that row every 2 tiles
        port_coord = []
        port_id = 0
        port_side_choices = [-1, 0, 1]

        port_side_prev = -1  # side of tile where the port on the previous tile in the line was
        for i in range(0, len(bottom_tiles)):
            # make sure there is no ports next to each other
            port_side = random.choice(port_side_choices)
            if port_side == 0:
                port_side_choices = [-1, 1]  # allow just one tile without ports
                port_side_prev = -1
                continue
            if port_side_prev == 1 and port_side == -1:
                port_side = 1
            port_side_prev = port_side

            x, y = bottom_tiles[i]
            x1 = x
            x2 = x + port_side
            y1 = float(y - hex_rad)
            y2 = float(y - hex_rad / 2)
            port_coord.append([x1, y1, x2, y2, port_id, ""])
            port_id += 1

            # we need only 5 ports
            if port_id == 5:
                break

        # Compute scaled y for tiles
        tile_centers = [(x, y, tile_id) for x, y, tile_id, *_ in double_coord]

        # For each port, find 5 closest tiles (or fewer), then pad with ID 20
        port_banned_tiles = []
        for port in port_coord:
            x1, y1, x2, y2, port_id, _ = port
            distances = []

            for tx, ty, tile_id in tile_centers:
                d = point_to_segment_distance(tx, ty, x1, y1, x2, y2)
                distances.append((d, tile_id))

            distances.sort()
            close_ids = [tile_id for _, tile_id in distances[:5]]

            # Pad if needed
            while len(close_ids) < 5:
                close_ids.append(20)

            port_banned_tiles.append([port_id] + close_ids)

        list_of_ports = ["wood", "wheat", "stone", "sheep", "brick"]
        default_port_locations = 0
        list_of_roll_numbers_start = [2, 3, 3, 3, 4, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 10, 11, 11, 12]
        list_of_tiles_start = [
            "sheep",
            "sheep",
            "sheep",
            "sheep",
            "wheat",
            "wheat",
            "wheat",
            "wheat",
            "wood",
            "wood",
            "wood",
            "wood",
            "brick",
            "brick",
            "brick",
            "brick",
            "stone",
            "stone",
            "stone",
            "stone",
            "stone",
        ]

    else:
        raise

    for p in port_coord:
        if default_port_locations == 1:
            p[5] = list_of_ports[p[4]]
        else:
            p[5] = random.choice(list_of_ports)
            list_of_ports.remove(p[5])

    return double_coord, port_coord, port_banned_tiles, list_of_ports, list_of_roll_numbers_start, list_of_tiles_start


def generate_board(version='original'):
    """
    Generates image of a balanced catan board
    :param version: original or got
    :return: image
    """

    num_of_tile_fails = 0
    num_of_number_fails = 0

    has_failed = 0
    successful_board = 1

    # config
    port_check = 1

    # get formations and config
    double_coord, port_coord, port_banned_tiles, list_of_ports, list_of_roll_numbers_start, list_of_tiles_start = get_formation(
        version)

    while successful_board == 1:
        # reset
        successful_board = 0
        has_failed = 0
        list_of_tiles = copy.deepcopy(list_of_tiles_start)

        # clear previous attempt decisions
        for c in double_coord:
            c[3] = ""

        for c in double_coord:
            # fix radius here
            if c[0] == 0 and c[1] == 0 and version == "original":
                c[3] = "desert"
            else:
                time_out_counter = 0
                if port_check == 1:
                    is_banned_by_port = 1
                    while is_banned_by_port == 1 and time_out_counter < 100:
                        tile_not_allowed = 0
                        # pick a tile resource
                        c[3] = random.choice(list_of_tiles)
                        # loop through the port IDs
                        for p in port_coord:
                            # check if the port type is equal to the selected tile resource
                            if p[5] == c[3]:
                                # check if the current hex ID is on the banned list for that port
                                for i in range(5):
                                    if port_banned_tiles[p[4]][i + 1] == c[2]:
                                        tile_not_allowed = 1
                                if tile_not_allowed == 0:
                                    is_banned_by_port = 0
                        time_out_counter = time_out_counter + 1
                else:
                    c[3] = random.choice(list_of_tiles)
                if time_out_counter >= 100:
                    successful_board = 1

                if len(list_of_tiles) != 1:
                    list_of_tiles.remove(c[3])

        # brick and stone check
        for c in double_coord:
            if c[3] == "stone" or c[3] == "brick":
                # run through tiles - if one is a neighbour then check if it is the same resource
                for d in double_coord:
                    if (
                        (d[0] == c[0] + 2 and d[1] == c[1])
                        or (d[0] == c[0] + 1 and d[1] == c[1] + 1)
                        or (d[0] == c[0] - 1 and d[1] == c[1] + 1)
                        or (d[0] == c[0] - 2 and d[1] == c[1])
                        or (
                            d[0] == c[0] - 1
                            and d[1] == c[1] - 1
                            or (d[0] == c[0] + 1 and d[1] == c[1] - 1)
                    )
                    ):
                        # print(str(d[2]) + " is a neighbour of " + str(c[2]))
                        if d[3] == c[3]:
                            has_failed = 1

        # other resources check
        for c in double_coord:
            if c[3] == "wheat" or c[3] == "wood" or c[3] == "sheep":
                # run through tiles - if one is a neighbour then check if it is the same resource
                for d in double_coord:
                    if (
                        (d[0] == c[0] + 2 and d[1] == c[1])
                        or (d[0] == c[0] + 1 and d[1] == c[1] + 1)
                        or (d[0] == c[0] - 1 and d[1] == c[1] + 1)
                        or (d[0] == c[0] - 2 and d[1] == c[1])
                        or (
                            d[0] == c[0] - 1
                            and d[1] == c[1] - 1
                            or (d[0] == c[0] + 1 and d[1] == c[1] - 1)
                    )
                    ):
                        # print(str(d[2]) + " is a neighbour of " + str(c[2]))
                        if d[3] == c[3]:
                            # if neighbour is the same resource check all of it's neighbours as well
                            for e in double_coord:
                                if (
                                    (e[0] == d[0] + 2 and e[1] == d[1])
                                    or (e[0] == d[0] + 1 and e[1] == d[1] + 1)
                                    or (e[0] == d[0] - 1 and e[1] == d[1] + 1)
                                    or (e[0] == d[0] - 2 and e[1] == d[1])
                                    or (
                                        e[0] == d[0] - 1
                                        and e[1] == d[1] - 1
                                        or (e[0] == d[0] + 1 and e[1] == d[1] - 1)
                                )
                                ):
                                    if e[2] != c[2] and e[3] == d[3]:
                                        has_failed = 1

        if has_failed == 1:
            has_failed = 0
            successful_board = 1
            # print("FAILED")
            num_of_tile_fails = num_of_tile_fails + 1

    # Assigning numbers
    successful_numbers = 1
    while successful_numbers == 1:
        # reset
        successful_numbers = 0
        list_of_roll_numbers = list_of_roll_numbers_start[:]
        has_failed_number = 0

        for c in double_coord:
            if c[0] == 0 and c[1] == 0:
                pass
            else:
                c[4] = random.choice(list_of_roll_numbers)
                if len(list_of_roll_numbers) != 1:
                    list_of_roll_numbers.remove(c[4])

        # check no two of the same number next to each other
        for c in double_coord:
            for d in double_coord:
                if (
                    (d[0] == c[0] + 2 and d[1] == c[1])
                    or (d[0] == c[0] + 1 and d[1] == c[1] + 1)
                    or (d[0] == c[0] - 1 and d[1] == c[1] + 1)
                    or (d[0] == c[0] - 2 and d[1] == c[1])
                    or (
                        d[0] == c[0] - 1
                        and d[1] == c[1] - 1
                        or (d[0] == c[0] + 1 and d[1] == c[1] - 1)
                )
                ):
                    # print(str(d[2]) + " is a neighbour of " + str(c[2]))
                    if d[4] == c[4]:
                        has_failed_number = 1

        # no two of the same number on one resource check
        for c in double_coord:
            for d in double_coord:
                if d[2] != c[2]:
                    if d[3] == c[3] and d[4] == c[4]:
                        has_failed_number = 1

        # no six and eight on the same resource
        for c in double_coord:
            if c[4] == 6 or c[4] == 8:
                for d in double_coord:
                    if d[2] != c[2]:
                        if d[3] == c[3] and (d[4] == 6 or d[4] == 8):
                            has_failed_number = 1

        # no six and eight next to eachother check
        for c in double_coord:
            if c[4] == 6 or c[4] == 8:
                for d in double_coord:
                    if (
                        (d[0] == c[0] + 2 and d[1] == c[1])
                        or (d[0] == c[0] + 1 and d[1] == c[1] + 1)
                        or (d[0] == c[0] - 1 and d[1] == c[1] + 1)
                        or (d[0] == c[0] - 2 and d[1] == c[1])
                        or (
                            d[0] == c[0] - 1
                            and d[1] == c[1] - 1
                            or (d[0] == c[0] + 1 and d[1] == c[1] - 1)
                    )
                    ):
                        # print(str(d[2]) + " is a neighbour of " + str(c[2]))
                        if d[4] == 6 or d[4] == 8:
                            has_failed_number = 1

        # no six and eight on bottom line for GOT version
        if version == 'got':
            for c in double_coord:
                if c[1] == 0 and c[4] in (6, 8):
                    has_failed_number = 1

        if has_failed_number == 1:
            has_failed_number = 0
            successful_numbers = 1
            # print("NUMBER FAILED")
            num_of_number_fails = num_of_number_fails + 1

    print(
        "Number of tile fails = "
        + str(num_of_tile_fails)
        + " and number of number fails = "
        + str(num_of_number_fails)
    )

    # Plotting
    color_scheme = {
        "sheep": "#C5E14C",  # soft green
        "wood": "#228B22",  # forest green
        "wheat": "#FCE618",  # warm yellow
        "brick": "#B22222",  # brick red
        "stone": "#545454",  # dark gray (stone)
        "desert": "#EDC9AF",
    }
    hex_rad = 2 / np.sqrt(3)
    ocean_alpha = 0.2
    if version == 'original':
        outer_coord = np.array([
            [2, 4 * hex_rad], [2, 5 * hex_rad], [-5 * (hex_rad / np.sqrt(3)), 5 * hex_rad],
            [-3 - (hex_rad / 2) * np.sqrt(3), 4 * hex_rad], [-3, 3.5 * hex_rad], [-2, 4 * hex_rad],
            [-1, 3.5 * hex_rad], [0, 4 * hex_rad], [1, 3.5 * hex_rad], [2, 4 * hex_rad]
        ])
        rotation_matrix = np.array([
            [math.cos(math.radians(60)), -math.sin(math.radians(60))],
            [math.sin(math.radians(60)), math.cos(math.radians(60))]
        ])
        # Plot all 6 rotated versions of the border
        coords = outer_coord.T
        for _ in range(6):
            xs, ys = coords
            plt.plot(xs, ys, color="blue", alpha=ocean_alpha)
            coords = rotation_matrix @ coords
    elif version == 'got':
        pass
    else:
        raise Exception("Unknown version")

    for c in double_coord:

        if c[3] in color_scheme:
            tile_colour = color_scheme[c[3]]
        else:
            tile_colour = "white"

        hexagon = RegularPolygon(
            (c[0], c[1] * 1.5 * hex_rad),
            numVertices=6,
            radius=hex_rad,
            alpha=0.7,
            edgecolor="k",
            facecolor=tile_colour,
        )
        ax.add_patch(hexagon)

    for p in port_coord:
        port_colour = color_scheme[p[5]] if p[5] in color_scheme else "black"
        circle_port = plt.Circle((p[0], p[1]), 0.2, edgecolor=port_colour, fill=False)
        ax.add_patch(circle_port)
        circle_port = plt.Circle((p[2], p[3]), 0.2, edgecolor=port_colour, fill=False)
        ax.add_patch(circle_port)

    # plot circles for numbers
    for c in double_coord:
        if c[0] == 0 and c[1] == 0:
            pass
        else:
            number_circle = plt.Circle(
                (c[0], c[1] * 1.5 * hex_rad), 0.4, edgecolor="black", facecolor="white"
            )
            ax.add_patch(number_circle)
            if c[4] == 6 or c[4] == 8:
                text_colour = "red"
            else:
                text_colour = "black"
            plt.text(
                c[0],
                c[1] * 1.5 * hex_rad,
                c[4],
                ha="center",
                va="center",
                size=8,
                color=text_colour,
            )

    plt.autoscale(enable=True)
    return plt.show()
