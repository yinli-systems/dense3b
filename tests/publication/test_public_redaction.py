"""Allow only explicitly audited public-byte replacements during archive rebuild."""
from pathlib import Path
import json
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import build_all_results as a

class PublicRedactionTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.rel='evidence/log.out'
        self.old=b'original archived diagnostic\n'
        self.new=b'published redacted diagnostic\n'
        path=self.root/self.rel;path.parent.mkdir(parents=True);path.write_bytes(self.new)
        self.r={'repo_path':self.rel,'sha256':a.sha(self.old),'stored_sha256':a.sha(self.old),'encoding':'identity'}
        self.entry={'path':self.rel,'original_sha256':a.sha(self.old),'public_sha256':a.sha(self.new),'public_bytes':len(self.new)}
        self.audit=self.root/'provenance/PUBLIC_NETWORK_REDACTION_AUDIT.json';self.audit.parent.mkdir()
        self.write_mapping([self.entry])
    def tearDown(self):self.tmp.cleanup()
    def write_mapping(self,entries):self.audit.write_text(json.dumps({'mapping':entries}))
    def test_accept_exact_recorded_public_bytes(self):
        self.assertEqual(a.raw(self.root,self.r),self.new)
        self.assertEqual(self.r['sha256'],a.sha(self.old))
    def test_reject_unrecorded_bytes(self):
        (self.root/self.rel).write_bytes(b'unapproved change')
        with self.assertRaises(ValueError):a.raw(self.root,self.r)
    def test_reject_wrong_original_hash(self):
        self.r['sha256']=a.sha(b'other original')
        with self.assertRaises(ValueError):a.raw(self.root,self.r)
    def test_reject_wrong_path_mapping(self):
        self.entry['path']='other/log.out';self.write_mapping([self.entry])
        with self.assertRaises(ValueError):a.raw(self.root,self.r)
    def test_reject_duplicate_mapping(self):
        self.write_mapping([self.entry,self.entry])
        with self.assertRaises(ValueError):a.raw(self.root,self.r)
    def test_reject_compressed_substitution(self):
        self.r['encoding']='gzip'
        with self.assertRaises(ValueError):a.raw(self.root,self.r)
    def test_reject_incorrect_public_size(self):
        self.entry['public_bytes']+=1;self.write_mapping([self.entry])
        with self.assertRaises(ValueError):a.raw(self.root,self.r)
    def test_accept_explicit_byte_identical_alias(self):
        alias=dict(self.entry,audited_path='audited/log.out')
        self.entry['path']='audited/log.out';self.write_mapping([self.entry])
        (self.audit.parent/'PUBLIC_NETWORK_REDACTION_ALIASES.json').write_text(json.dumps({'mapping':[alias]}))
        self.assertEqual(a.raw(self.root,self.r),self.new)
    def test_reject_alias_without_matching_anchor(self):
        alias=dict(self.entry,audited_path='missing-anchor')
        self.entry['path']='audited/log.out';self.write_mapping([self.entry])
        (self.audit.parent/'PUBLIC_NETWORK_REDACTION_ALIASES.json').write_text(json.dumps({'mapping':[alias]}))
        with self.assertRaises(ValueError):a.raw(self.root,self.r)
    def test_reject_absent_audit(self):
        self.audit.unlink()
        with self.assertRaises(ValueError):a.raw(self.root,self.r)
    def test_reject_path_traversal(self):
        self.r['repo_path']='../outside'
        with self.assertRaises(ValueError):a.raw(self.root,self.r)
    def test_reject_symlink_outside_root(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'outside';p.write_bytes(self.new)
            link=self.root/'link';link.symlink_to(p)
            self.r['repo_path']='link'
            with self.assertRaises(ValueError):a.raw(self.root,self.r)

if __name__=='__main__':unittest.main()
