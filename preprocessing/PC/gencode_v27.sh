
version=27
cd /home/data/tangxl/reference/ || exit
mkdir gencode_v${version}
cd gencode_v${version} || exit

wget http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${version}/gencode.v${version}.2wayconspseudos.gtf.gz
wget http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${version}/gencode.v${version}.long_noncoding_RNAs.gtf.gz 
wget http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${version}/gencode.v${version}.polyAs.gtf.gz 
wget http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${version}/gencode.v${version}.annotation.gtf.gz

wget http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${version}/GRCh37_mapping/gencode.v${version}lift37.annotation.gtf.gz 
wget http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${version}/GRCh37_mapping/gencode.v${version}lift37.metadata.HGNC.gz 
wget http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${version}/GRCh37_mapping/gencode.v${version}lift37.metadata.EntrezGene.gz 
wget http://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_${version}/GRCh37_mapping/gencode.v${version}lift37.metadata.RefSeq.gz 


mkdir gene_position
zcat  gencode.v${version}.long_noncoding_RNAs.gtf.gz |perl -alne '{next unless $F[2] eq "gene" ;/gene_name \"(.*?)\";/; print "$F[0]\t$F[3]\t$F[4]\t$1" }' >./gene_position/lncRNA.hg38.position
zcat  gencode.v${version}.2wayconspseudos.gtf.gz     |perl -alne '{next unless $F[2] eq "transcript" ;/gene_name \"(.*?)\";/; print "$F[0]\t$F[3]\t$F[4]\t$1" }' >./gene_position/pseudos.hg38.position
zcat  gencode.v${version}.annotation.gtf.gz| grep   protein_coding |perl -alne '{next unless $F[2] eq "gene" ;/gene_name \"(.*?)\";/; print "$F[0]\t$F[3]\t$F[4]\t$1" }' >./gene_position/protein_coding.hg38.position
zcat  gencode.v${version}.annotation.gtf.gz|perl -alne '{next unless $F[2] eq "gene" ;/gene_name \"(.*?)\";/; print "$F[0]\t$F[3]\t$F[4]\t$1" }' >./gene_position/allGene.hg38.position
 
zcat  gencode.v${version}lift37.annotation.gtf.gz | grep   protein_coding |perl -alne '{next unless $F[2] eq "gene" ;/gene_name \"(.*?)\";/; print "$F[0]\t$F[3]\t$F[4]\t$1" }' >./gene_position/protein_coding.hg19.position
zcat  gencode.v${version}lift37.annotation.gtf.gz | perl -alne '{next unless $F[2] eq "gene" ;/gene_name \"(.*?)\";/; print "$F[0]\t$F[3]\t$F[4]\t$1" }' >./gene_position/allGene.hg19.position
