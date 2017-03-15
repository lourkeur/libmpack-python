"""hypothesis strategies for testing libmpack.

Each strategy's goal is to generate a value suitable for a particular format,
pack it using a slow but easily verified implementation of MessagePack, and
then return the packed representation tupled with the original value.
"""

from hypothesis.strategies import (none, booleans, integers, lists,
        dictionaries, recursive, one_of, text, binary)
from hypothesis.searchstrategy import SearchStrategy
import numpy


class StrategyProxy(SearchStrategy):
    def __init__(self, st):
        self.__st = st

    def __getattr__(self, name):
        return getattr(self.__st, name)

    def do_draw(self, *args, **kwargs):
        return self.__st.do_draw(*args, **kwargs)

    def __or__(self, other):
        return one_of(self, other)

    def __repr__(self):
        return repr(self.__st)

class StrategyAlias(StrategyProxy):
    def __init__(self, name, *args, **kwargs):
        self.__name = name
        super(StrategyAlias, self).__init__(*args, **kwargs)

    def __repr__(self):
        return self.__name

_all_scalar = []

def _scalar_st(name, source, pack, postpack=lambda v: v):
    st = StrategyAlias(u"mpack.%s" % name,
            source.map(lambda v: (pack(v), postpack(v))))
    _all_scalar.append(st)
    globals()[name] = st

_scalar_st(u"nil", none(), lambda v: b"\xc0")
_scalar_st(u"bool", booleans(), lambda v: b"%c" % (0xc2 + v))
_scalar_st(u"positive_fixnum", integers(0, 127),
        lambda v: b"%c" % v)
_scalar_st(u"negative_fixnum", integers(-32, -1),
        lambda v: b"%c" % (v + 256 | 0xe0))

def _num_st(name, firstbyte, dtype, postpack=lambda x: x):
    from hypothesis.extra.numpy import arrays
    # arrays(dtype, ()) yields integer without endianess metadata
    _scalar_st(name, arrays(dtype, 1),
            pack=lambda v: b"%c%s" % (firstbyte, v.tobytes()),
            postpack=lambda v: postpack(v[0]))

for i in range(4):
    _num_st(u"int%d" % (8 * 2**i), 0xd0 + i, ">i%d" % 2**i)
    _num_st(u"uint%d" % (8 * 2**i), 0xcc + i, ">u%d" % 2**i)

class _Nan(object):
    """It's equal to this number that's not even equal to itself.

    This greatly simplifies things downstream.
    """
    def __eq__(self, other):
        if not numpy.isreal(other):
            return NotImplemented
        return numpy.isnan(other)

    def __ne__(self, other):
        if not numpy.isreal(other):
            return NotImplemented
        return not numpy.isnan(other)

    def __hash__(self):
        return hash(numpy.nan)

_nan = _Nan()
def _float_postpack(v):
    return _nan if numpy.isnan(v) else v

_num_st(u"float32", 0xca, ">f4", postpack=_float_postpack)
_num_st(u"float64", 0xcb, ">f8", postpack=_float_postpack)


def _int_tobytes(v, nbytes):
    return numpy.array(v, ">u%d" % nbytes).tobytes()

def _pack_bin(firstbyte, nbytes_len):
    def pack(v):
        _len = _int_tobytes(len(v), nbytes_len)
        return b"%c%s%s" % (firstbyte, _len, v)
    return pack

for i in range(3):
    _scalar_st(u"bin%d" % (8 * 2**i), binary(max_size=2**(8 * 2**i) - 1, average_size=20),
            pack=_pack_bin(0xc4 + i, 2**i))


def _text(max_size):
    """makes unicode string with an utf-8 length of at most max_size.

    `hypothesis.strategies.text`'s `max_size` counts unicode *characters*, but we
    can't go over a certain number of *bytes*, so this wrapper makes up for it.
    `text` is used internally as an approximate.
    """
    return text(max_size=max_size, average_size=20).filter(lambda v: len(v.encode()) <= max_size)

_scalar_st(u"fixstr", _text(max_size=31),
        pack=lambda v: (b"%c%s" % (0xa0 | len(v.encode()), v.encode())))

def _pack_str(i):
    pack2 = _pack_bin(0xd9 + i, 2**i)
    def pack(v):
        return pack2(v.encode())
    return pack

for i in range(3):
    _scalar_st(u"str%d" % (8 * 2**i), _text(max_size=2**(8 * 2**i) - 1),
            pack=_pack_str(i))


all_scalar = one_of(_all_scalar)


def _concat_elements(l):
    if len(l) == 0:
        return b"", []
    else:
        packed_vs, vs = zip(*l)
        return b"".join(packed_vs), list(vs)


class StrategyWrapper(StrategyAlias):
    def __init__(self, params, *args, **kwargs):
        self.__params = params.copy()
        source = self._from_params(params)
        super(StrategyWrapper, self).__init__(st=self._make_st(source), *args, **kwargs)

    def with_params(self, **kwargs):
        if not kwargs:
            return self
        params = self.__params.copy()
        params.update(kwargs)
        return type(self)(params)

    def with_source(self, source):
        name = u"%s.with_source(%s)" % (self, source)
        st = self._make_st(self._from_alternate_source(source))
        return StrategyAlias(name, st)

    def __repr__(self):
        name = super(StrategyWrapper, self).__repr__()
        params = self.__params
        if params:
            return u"%s.with_params(%s)" % (name, ", ".join(u"%s=%s" % i for i in sorted(params.items())))
        else:
            return name

_all_arrays = []
def _array_st(name, max_size, pack, postpack=lambda x: x):
    def pack2(l):
        packed_l, l = _concat_elements(l)
        return pack(packed_l, l), postpack(l)

    class st(StrategyWrapper):
        def __init__(self, params):
            super(st, self).__init__(params=params, name=name)

        @staticmethod
        def _make_st(source):
            return source.map(pack2)

        @staticmethod
        def _from_params(params):
            try:
                params["max_size"] = min(params["max_size"], max_size)
            except KeyError:
                params["max_size"] = max_size
            params.setdefault("elements", all_scalar)
            return lists(**params)

        @staticmethod
        def _from_alternate_source(source):
            return source.filter(lambda l: len(l) <= max_size)

    singleton = st(params={"average_size": min(20, max_size)})
    _all_arrays.append(singleton)
    globals()[name] = singleton

_array_st(u"fixarray",
        max_size=15,
        pack=lambda packed_l, l: b"%c%s" % (0x90 | len(l), packed_l))

def _pack_array(firstbyte, nbytes_len):
    def pack(packed_l, l):
        _len = _int_tobytes(len(l), nbytes_len)
        return b"%c%s%s" % (firstbyte, _len, packed_l)
    return pack

_array_st(u"array16", max_size=2**16 - 1, pack=_pack_array(0xdc, 2))
_array_st(u"array32", max_size=2**32 - 1, pack=_pack_array(0xdd, 4))

all_arrays = one_of(_all_arrays)

all_hashable_types = recursive(
        base=all_scalar,
        extend=lambda S: one_of(a.with_params(elements=S) for a in _all_arrays), max_leaves=5)


def _concat_items(d):
    packed_items, items = _concat_elements(
            [(packed_key + packed_val, (key, val))
                for (packed_key, key), (packed_val, val) in d.items()])
    return packed_items, dict(items)


_all_maps = []
def _map_st(name, max_size, pack, postpack=lambda x: x):
    def pack2(l):
        packed_d, d = _concat_items(l)
        return pack(packed_d, d), postpack(d)

    class st(StrategyWrapper):
        def __init__(self, params):
            super(st, self).__init__(params=params, name=name)

        @staticmethod
        def _make_st(source):
            return source.map(pack2)

        @staticmethod
        def _from_params(params):
            try:
                params["max_size"] = min(params["max_size"], max_size)
            except KeyError:
                params["max_size"] = max_size
            params.setdefault("keys", all_scalar)
            params.setdefault("values", all_scalar)
            return dictionaries(**params)

        @staticmethod
        def _from_alternate_source(source):
            return source.filter(lambda d: len(d) <= max_size)

    singleton = st(params={"average_size": min(20, max_size)})
    _all_maps.append(singleton)
    globals()[name] = singleton

_map_st(u"fixmap",
        max_size=15,
        pack=lambda packed_d, d: b"%c%s" % (0x80 | len(d), packed_d))


_pack_map = _pack_array
_map_st(u"map16", max_size=2**16 - 1, pack=_pack_map(0xde, 2))
_map_st(u"map32", max_size=2**32 - 1, pack=_pack_map(0xdf, 4))

all_maps = one_of(_all_maps)
all_types = recursive(
        base=all_hashable_types,
        extend=lambda S: one_of(m.with_params(values=S) for m in _all_maps), max_leaves=5)
