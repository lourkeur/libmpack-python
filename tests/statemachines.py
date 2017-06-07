from hypothesis.stateful import RuleBasedStateMachine, Bundle, rule

import collections
import mpack
import unittest

from .strategies import *


class PendingRequestData(object):
    pass

class PendingRequest(object):
    def __init__(self, msgid, data, pending):
        self.msgid, self.data, self.pending = msgid, data, pending


def msg_tail(t, **kwargs):
    return msg_contents(msg_types(just(t)), msgids=positive_fixnum(just(0)), **kwargs).map(lambda msg: msg[1:])

class RPCSession(RuleBasedStateMachine):
    pending_requests = Bundle("pending requests")

    def __init__(self):
        super().__init__()
        self.session = mpack.Session()

    @rule(target=pending_requests, x=msg_tail('request'))
    def do_request(self, x):
        _, method, params = x
        data = PendingRequestData()
        msg = self.session.request(
                mpack.unpack(method.packed),  # can't pack nonstandard classes like nan and numpy.*
                mpack.unpack(params.packed),  # so here's a simple and effective fix.
                data=data)
        contents = mpack.unpack(msg)
        assert contents[0] == MSG_TYPES.index('request')
        msgid = contents[1]
        assert contents[2] == method.orig
        assert contents[3] == params.orig
        return PendingRequest(msgid, data, True)

    @rule(req=pending_requests, rep=msg_tail('response'))
    def eat_response(self, req, rep):
        assume(req.pending)
        req.pending = False
        _, error, result = rep
        msg = mpack.pack([
            MSG_TYPES.index('response'),
            req.msgid,
            mpack.unpack(error.packed),
            mpack.unpack(result.packed),
            ])
        assert self.session.receive(msg) == (len(msg), 'response', error, result, req.data)

    @rule(x=msg(msg_contents(msg_types(sampled_from(('request', 'notification'))))))
    def eat_request_or_notification(self, x):
        t = x.orig[0]
        msgid = x.orig[1] if t == 'request' else None
        method, params = x.orig[-2:]
        assert self.session.receive(x.packed) == (len(x.packed), t, method, params, msgid)

    @rule(x=msg_tail('notification'))
    def do_notification(self, x):
        method, params = x
        msg = self.session.notify(*(mpack.unpack(el.packed) for el in x))
        contents = mpack.unpack(msg)
        assert contents[0] == MSG_TYPES.index('notification')
        assert contents[1] == method.orig
        assert contents[2] == params.orig

    @rule(x=msg_tail('response'))
    def do_response(self, x):
        msgid, error, result = x
        has_err = error.orig is not None
        y = msgid, error if has_err else result
        msg = self.session.reply(*(mpack.unpack(el.packed) for el in y), has_err)
        contents = mpack.unpack(msg)
        assert contents[0] == MSG_TYPES.index('response')
        assert contents[1] == msgid.orig
        assert contents[2] == error.orig
        assert contents[3] == result.orig
