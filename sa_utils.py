import itertools
import pickle

# --- Discrete Action Space ---
MOVE_ACTIONS = ['S', 'L', 'R', 'U', 'D']
PKG_ACTIONS = ['0', '1', '2']
# Action space: Cartesian product of move and package actions
ACTION_SPACE = [(m, p) for m in MOVE_ACTIONS for p in PKG_ACTIONS]
# Mappings between action tuples and indices
ACTION2IDX = {action: idx for idx, action in enumerate(ACTION_SPACE)}
IDX2ACTION = {idx: action for idx, action in enumerate(ACTION_SPACE)}


def encode_action_tuple(move, pkg_act):
    """Chuyển tuple (move, pkg_act) thành chỉ số rời rạc."""
    return ACTION2IDX[(move, pkg_act)]


def decode_action_idx(idx):
    """Chuyển chỉ số rời rạc thành tuple (move, pkg_act)."""
    return IDX2ACTION[idx]


class StateEncoder:
    """
    Class để mã hóa state (dictionary) thành index rời rạc và ngược lại.
    Sử dụng pickle để đảm bảo key hashable.
    """
    def __init__(self):
        self.state2idx = {}
        self.idx2state = {}

    def encode(self, state):
        """Mã hóa state (ví dụ dict) thành index int."""
        key = pickle.dumps(state)
        if key not in self.state2idx:
            idx = len(self.state2idx)
            self.state2idx[key] = idx
            self.idx2state[idx] = state
        return self.state2idx[key]

    def decode(self, idx):
        """Lấy lại state (dictionary) từ index."""
        return self.idx2state[idx]

    def all_states(self):
        """Trả về danh sách tất cả state đã mã hóa."""
        return list(self.idx2state.items()) 