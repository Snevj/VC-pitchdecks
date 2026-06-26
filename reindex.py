#it is needed when using evals because byoasses the need for using frontend


from src.pipeline import index_pdf_week2

# index_pdf_week2 automatically handles clear_collection() natively!
index_pdf_week2("uploads/nestle_files.pdf")
print("🎉 Done — Qdrant now has fresh Nestle chunks!")