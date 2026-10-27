#!/bin/bash
# The one WikiCoach crawl chain. Refuses to start twice.
cd "$(dirname "$0")"
LOCK=../data/wikicoach/.crawl.lock
if [ -f $LOCK ] && kill -0 "$(cat $LOCK)" 2>/dev/null; then echo "already running"; exit 1; fi
echo $$ > $LOCK
python build_wikicoach.py revs > ../logs/wikicoach_revs.log 2>&1
python build_wikicoach.py revs >> ../logs/wikicoach_revs.log 2>&1
python build_wikicoach.py content > ../logs/wikicoach_content.log 2>&1 &&
python build_wikicoach.py resolve > ../logs/wikicoach_resolve.log 2>&1 &&
python build_wikicoach.py claims > ../logs/wikicoach_claims.log 2>&1
tail -n 2 ../logs/wikicoach_claims.log
rm -f $LOCK
