# Python-Based Physical Uncloneable Function Emulation
Implementation of a PUF emulator along with a ML and DL algorithm to attempt to imulate the PUF. 

## Description
We want to implement a software model (an emulator) of a Physical Unclonable Function (PUF). A PUF is essentially a “digital fingerprint” for hardware, producing unique outputs when given specific inputs because of tiny and uncontrollable differences in the way each chip is manufactured. (van der Leest). They are a crucial element of modern hardware security that ensures silicon devices can be authenticated for physical supply chain distribution, tampering detection, and Secure Boot processes. 

While intrinsically rooted in hardware, our Python implementation will simplify the process and enable us to quickly generate large sets of challenge-response pairs (inputs and outputs) to evaluate their quality. Quality will be measured using key metrics such as reliability (whether the same chips give the same answer to the same input consistently), and uniqueness (whether different chips produce different responses). We will then train a simple machine learning model to see if it is able to predict the emulator’s outputs based on its inputs. This will simulate the type of attack that could possibly compromise a PUF in a real-world setting and allow a hacker to impersonate a computing device. By analyzing the success or failure of this attack, we will better understand both the strengths and weaknesses of PUFs and how they fit into the modern cybersecurity landscape.

We will all take ownership over the research, programming, and writing aspects of the project to allow us all to benefit from learning and implementing the technical skills. However, we will assign ourselves roles in becoming “experts” in the python libraries that we will be using (PyPUF, MatPlotLib, Pandas, and PyTorch) so that we can more efficiently identify the components of the libraries that are necessary for our project. After completing our individual research on the topics, tasks will be delegated according to skills in an even distribution for a fair split of work on October 13th, 2025. 


## Getting Started
How does someone go about using our project?
### Dependencies
What libraries and other prerequisites will users need before using our code?
### Executing Programming
How would someone set up and use our implementations?

## Authors
Anthony C.

Eli G.

Adya S.

Nate W. - <nww831@uw.edu>

## Version History
Any updates we make

## Acknowledgments
Any external acknowledgements and citations we may have
