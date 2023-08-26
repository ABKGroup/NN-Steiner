# def _terminals2vec(self, cell: Cell, qt: QuadTree) -> np.ndarray:
#         terminals_id: List[int] = cell.terminals_id
#         ret_list: List[List[float]] = []
#         for terminal_id in terminals_id:
#             terminal: Terminal = qt.terminals[terminal_id]
#             (x, y) = self._point_norm2cell(terminal, cell)
#             ret_list.append([x, y])
#         for _ in range(len(terminals_id), qt.args["kb"]):
#             # ret_list.append([0, 0])
#             ret_list.append([-1, -1])

#         ret: np.ndarray = np.array(ret_list)
#         assert ret.shape == (qt.args["kb"], 2)
#         return ret

# def _sides2vec_tfm(self, cell: Cell, side_typ: str, qt: QuadTree) -> np.ndarray:
#         """
#         side_typ: "boundary" or "cross"
#         sides: [right, bottom, left, top]
#             side: [portal * num_side_portal]
#                 point: [x, y] (can be null)
#             shape: (4 * num_side_portal,2)
#         """
#         # portal(2) * (m+2) * sides(4)
#         ret_list: List[List[float]] = []

#         def update_sides(sides: List[Side]) -> None:
#             for side in sides:
#                 ret_list.extend(self._side2vec_tfm(side, qt))

#         # sides
#         if side_typ == "boundary":
#             update_sides(
#                 [
#                     qt.sides[cell.boundary.east],
#                     qt.sides[cell.boundary.south],
#                     qt.sides[cell.boundary.west],
#                     qt.sides[cell.boundary.north],
#                 ]
#             )
#         elif side_typ == "cross":
#             assert cell.cross is not None, cell.id
#             update_sides(
#                 [
#                     qt.sides[cell.cross.right],
#                     qt.sides[cell.cross.bottom],
#                     qt.sides[cell.cross.left],
#                     qt.sides[cell.cross.top],
#                 ]
#             )

#         ret: np.ndarray = np.array(ret_list)
#         assert ret.shape == (4 * qt.n_side_p, 2), ret
#         return ret

# def _side2vec_tfm(self, side: Side, qt: QuadTree) -> np.ndarray:
#         """
#         point: [x, y] (normalized, could be null)
#         null: [-1, -1]
#         """
#         side_ret_list: List[List[float]] = []
#         for portal_id in side.portals_id:
#             if portal_id is None:
#                 # side_ret_list.append([0, 0])
#                 side_ret_list.append([-1., -1.])
#             else:
#                 portal: Portal = qt.portals[portal_id]
#                 (x, y) = self._point_norm2cell(portal, qt.cells[side.cell_id])
#                 side_ret_list.append([x, y])
#         assert len(side_ret_list) == qt.n_side_p
#         return np.array(side_ret_list)
