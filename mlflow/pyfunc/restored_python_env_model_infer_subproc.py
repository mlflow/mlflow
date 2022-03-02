import socket
import sys
import importlib
import argparse
import pickle

_PICKLE_PROTOCOL_FOR_RESTORE_PY_ENV = 3

if __name__ == "__main__":
    sys.stdin.close()  # for safety, close it.
    parser = argparse.ArgumentParser()
    parser.add_argument('--server_port', type=int)
    parser.add_argument('--loader_module', type=str)
    parser.add_argument('--model_path', type=str)

    args = parser.parse_args()

    model_impl = importlib.import_module(args.loader_module)._load_pyfunc(args.model_path)

    sock = socket.socket()
    try:
        sock.connect(('127.0.0.1', args.server_port))
        comm_stream = sock.makefile('rwb')

        while True:
            batch = pickle.load(comm_stream)
            if batch is None:
                break

            # TODO: pipeline IO and inference computation
            infer_result_batch = model_impl.predict(batch)
            print('subproc predict batch done.')
            pickle.dump(infer_result_batch, comm_stream, protocol=_PICKLE_PROTOCOL_FOR_RESTORE_PY_ENV)
            comm_stream.flush()

        # write a None object as the data end signal.
        pickle.dump(None, comm_stream, protocol=_PICKLE_PROTOCOL_FOR_RESTORE_PY_ENV)
        comm_stream.flush()
    finally:
        comm_stream.close()
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()
