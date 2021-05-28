ARG TAG

FROM deepclr-deps:${TAG}

# Install DeepCLR
COPY LICENSE README.md requirements.txt setup.cfg setup.py /tmp/deepclr/
COPY deepclr /tmp/deepclr/deepclr

RUN cd /tmp/deepclr \
		&& python setup.py install \
		&& rm -rf /tmp/deepclr
