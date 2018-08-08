# Import a dependency in MLflow's setup.py that's not included by default in conda environments,
# verify that it fails


def main():
    try:
        import gunicorn
    except ImportError:
        print("Import of gunicorn failed as expected")
        return
    raise Exception("Expected exception when attempting to import gunicorn.")


if __name__ == "__main__":
    main()
