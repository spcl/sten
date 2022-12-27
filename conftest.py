import sten


def pytest_runtest_setup(item):
    # some test may forget to return this global flag to warning mode,
    # so we need to return it to original state before running each test.
    sten.set_dispatch_failure(sten.DISPATCH_WARN)
