from apps.hello import hello

def test_hello():
    assert hello("test") == f"Hello test!"