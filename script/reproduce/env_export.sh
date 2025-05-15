# 1️⃣ capture conda‑explicit packages only
conda env export --from-history | grep -v prefix: > environment.yml

# 2️⃣ append only packages whose installer == 'pip'
pip list --format=json \
  | python -c "import sys, json; pkgs=json.load(sys.stdin); \
               print('- pip'); print('  - pip:'); \
               [print(f'      - {p['name']}=={p['version']}') \
                for p in pkgs if p.get('installer')=='pip']" \
  >> environment.yml
