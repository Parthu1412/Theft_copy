import ssl


def create_ssl_context():
    _ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    _ssl_context.options |= ssl.OP_NO_SSLv2
    _ssl_context.options |= ssl.OP_NO_SSLv3
    _ssl_context.check_hostname = False
    _ssl_context.verify_mode = ssl.CERT_NONE
    _ssl_context.load_default_certs()

    return _ssl_context
